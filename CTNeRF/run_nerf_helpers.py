import os
import torch
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

from torchvision.models import resnet34, ResNet34_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Misc utils
def img2mse(x, y, M=None):
    if M == None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x - y) ** 2 * M) / (torch.sum(M) + 1e-8) / x.shape[-1]

def CharbonnierLoss(x, y):
    return torch.mean(torch.sqrt(torch.pow((x - y), 2) + 1e-8))

def img2mae(x, y, M=None):
    if M == None:
        return torch.mean(torch.abs(x - y))
    else:
        return torch.sum(torch.abs(x - y) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def L1(x, M=None):
    if M == None:
        return torch.mean(torch.abs(x))
    else:
        return torch.sum(torch.abs(x) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def L2(x, M=None):
    if M == None:
        return torch.mean(x ** 2)
    else:
        return torch.sum((x ** 2) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-19)) / x.shape[0]


def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']  # 4
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)  # add a anonymous function x to the list
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim



class PixelNeRF_d(nn.Module):
    def __init__(self, f1D=3, f2D=2, W=256, input_ch=3, input_ch_views=3, output_ch=4, img_fea_ch=None,
                 use_viewdirsDyn=True):
        super(PixelNeRF_d, self).__init__()

        self.f1D = f1D
        self.f2D = f2D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirsDyn = use_viewdirsDyn
        self.img_fea_ch = img_fea_ch

        self.pts_mlp = nn.Sequential(
            nn.Linear(self.input_ch, W),
            nn.ReLU(inplace=True)
        )

        f1 = []
        f2 = []

        for i in range(self.f1D):
            f1.append(ResMLP(self.W, img_fea_ch, is_dynamic=True))

        for i in range(self.f2D):
            f2.append(ResMLP(self.W))

        self.f1 = nn.ModuleList(f1)
        self.f2 = nn.ModuleList(f2)

        self.weight_linear = nn.Linear(self.W, 1)
        self.fc_linear = nn.Sequential(
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True)
        )
        self.sf_linear = nn.Linear(W, 6)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if self.use_viewdirsDyn:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.final_layer = nn.Linear(self.W, output_ch)

        self.Cross_Attention = Cross_Attention(self.W, self.img_fea_ch)
        self.Cross_Attention_n = Cross_Attention_n(self.W)

        self.GlobalFilter = GlobalFilter()
        self.feature_fc = nn.Linear(img_fea_ch, W)


    def forward(self, x, img_feature):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        img_feature = img_feature.unsqueeze(0).reshape(img_feature.shape[0] // input_pts.shape[0], -1,
                                                       img_feature.shape[-1])

        t = self.Cross_Attention(img_feature[:1, ...], img_feature[1:, ...], img_feature[1:, ...])
        t = self.Cross_Attention_n(t, t, t)

        img_feature = self.feature_fc(img_feature)
        img_feature = self.GlobalFilter(img_feature)

        t = img_feature[1:, ...] + t
        # t = img_feature[1:, ...]

        t = t.mean(0)

        h = self.pts_mlp(input_pts)

        for layer in self.f1:
            h = layer(h, t)

        # h = h.mean(0)

        for layer in self.f2:
            h = layer(h)

        sf_b = self.fc_linear(h)
        sf = torch.tanh(self.sf_linear(sf_b))
        blending = torch.sigmoid(self.weight_linear(sf_b))

        if self.use_viewdirsDyn:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.final_layer(h)

        return torch.cat([outputs, sf, blending], -1)


class PixelNeRF_s(nn.Module):
    def __init__(self, f1D=3, f2D=2, W=256, input_ch=3, input_ch_views=3, output_ch=4, img_fea_ch=None,
                 use_viewdirs=True):
        super(PixelNeRF_s, self).__init__()

        self.f1D = f1D
        self.f2D = f2D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs

        self.pts_mlp = nn.Sequential(
            nn.Linear(self.input_ch, W),
            nn.ReLU(inplace=True)
        )

        f1 = []
        f2 = []

        for i in range(self.f1D):
            f1.append(ResMLP(self.W, img_fea_ch))

        for i in range(self.f2D):
            f2.append(ResMLP(self.W))

        self.f1 = nn.ModuleList(f1)
        self.f2 = nn.ModuleList(f2)

        self.weight_linear = nn.Linear(self.W, 1)

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        if self.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.final_layer = nn.Linear(self.W, output_ch)

    def forward(self, x, img_feature):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        img_feature = img_feature.unsqueeze(0).reshape(img_feature.shape[0] // input_pts.shape[0], -1,
                                                       img_feature.shape[-1])


        h = self.pts_mlp(input_pts)
        for layer in self.f1:
            h = layer(h, img_feature)

        h = h.mean(dim=0)

        for layer in self.f2:
            h = layer(h)
        blending = torch.sigmoid(self.weight_linear(h))

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h, inplace=True)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.final_layer(h)

        return torch.cat([outputs, blending], -1)

class GlobalFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.complex_weight = None

    def forward(self, x):
        B, dim_feature, C = x.shape
        x = torch.fft.rfft2(x, dim=(0,1), norm='ortho')
        if self.complex_weight is None:
            self.complex_weight = nn.Parameter(torch.randn(B, x.shape[1], C, 2, dtype=torch.float32) * 0.02)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(B,dim_feature), dim=(0,1), norm='ortho')
        return x

class ResMLP(nn.Module):
    def __init__(self, hidden_ch, img_fea_ch=None, is_dynamic=False):
        super(ResMLP, self).__init__()
        if img_fea_ch is not None:
            if is_dynamic:
                self.img_mlp = nn.Sequential(
                    nn.Linear(hidden_ch, hidden_ch),
                    nn.ReLU(inplace=True),
                )
            else:
                self.img_mlp = nn.Sequential(
                    nn.Linear(img_fea_ch, hidden_ch),
                    nn.ReLU(inplace=True),
                )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, res_input, img_fea_input=None):
        if img_fea_input is not None:
            res_input = res_input + self.img_mlp(img_fea_input)
        return self.mlp(res_input) + res_input

class Cross_Attention(nn.Module):
    def __init__(self, dim, feature_dim):
        super(Cross_Attention, self).__init__()
        self.q_fc = nn.Linear(feature_dim, dim, bias=False)
        self.k_fc = nn.Linear(feature_dim, dim, bias=False)
        self.v_fc = nn.Linear(feature_dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.act_fun = nn.GELU()

    def forward(self, q, k, v):
        views, _, _ = k.shape
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(v)

        q = q.repeat(k.shape[0] // q.shape[0], 1, 1)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).contiguous()
        x = self.act_fun(self.out_fc(out)) + q
        del attn, out, q, k, v

        return x.view(views, -1, x.shape[-1])


class Cross_Attention_n(nn.Module):
    def __init__(self, dim):
        super(Cross_Attention_n, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.act_fun = nn.GELU()
        self.ray_sample = 64
        self.pos_encoding = self.posenc(d_hid=dim, n_samples=self.ray_sample)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
            # return [position * 2. ** hid_j // 25.6 for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, q, k, v):
        views, _, _ = k.shape
        residual = q
        residual = residual.view(-1, self.ray_sample, q.shape[-1])
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(v)

        q = q.view(-1, self.ray_sample, q.shape[-1])
        k = k.view(-1, self.ray_sample, k.shape[-1])
        v = v.view(-1, self.ray_sample, v.shape[-1])

        q = q + self.pos_encoding.to(q.device)
        k = k + self.pos_encoding.to(k.device)
        v = v + self.pos_encoding.to(k.device)

        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).contiguous()
        x = self.act_fun(self.out_fc(out)) + residual
        del attn, out, q, k, v, residual

        return x.view(views, -1, x.shape[-1])


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        feats3 = self.resnet.layer2(feats2)
        feats4 = self.resnet.layer3(feats3)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )

        latents = torch.cat(latents, dim=1)
        return latents


class Reference:
    def __init__(self, reference, c2w, f, img_size, K=None):
        self.reference = reference
        self.scale = (img_size / 2) / f
        self.n = c2w.shape[0]
        self.R_t = c2w[:, :3, :3].permute(0, 2, 1).clone().detach().to(reference.device)
        self.camera_pos = c2w[:, :3, -1].clone().detach().to(reference.device)

    @torch.no_grad()
    def feature_matching(self, pos):
        n_rays, n_samples, _ = pos.shape
        pos = pos.unsqueeze(dim=0).expand([self.n, n_rays, n_samples, 3])
        camera_pos = self.camera_pos[:, None, None, :]
        camera_pos = camera_pos.expand_as(pos)
        ref_pos = torch.einsum("kij,kbsj->kbsi", self.R_t, pos - camera_pos)
        uv_pos = ref_pos[..., :-1] / ref_pos[..., -1:] / self.scale
        uv_pos[..., 1] *= -1.0
        return F.grid_sample(self.reference, uv_pos, align_corners=True, padding_mode="border")


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs, f_projection=None):
        if f_projection is not None:
            if f_projection.shape[0] == 3:
                return torch.cat([fn(inputs[i:i + chunk],
                                     torch.cat([f_projection[0][i:i + chunk], f_projection[1][i:i + chunk],
                                                f_projection[2][i:i + chunk]],
                                               dim=0)) for i in range(0, inputs.shape[0], chunk)], 0)
            else:
                return torch.cat([fn(inputs[i:i + chunk],
                                     torch.cat([f_projection[0][i:i + chunk], f_projection[1][i:i + chunk]],
                                               dim=0)) for i in range(0, inputs.shape[0], chunk)], 0)
        else:
            return torch.cat(
                [fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64, f_projection=None):
    """Prepares inputs and applies network 'fn'.
    """

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs[:, :, :3].shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    if f_projection is not None:
        B, C, N_rays, N_samples = f_projection.shape
        f_projection = f_projection.permute(0, 2, 3, 1)
        f_projection = f_projection.reshape(B, N_rays * N_samples, C)
        outputs_flat = batchify(fn, netchunk)(embedded, f_projection)  # net input
    else:
        outputs_flat = batchify(fn, netchunk)(embedded)  # net input

    outputs = torch.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """

    encoder = ImageEncoder().to(device).eval()

    output_ch = 5 if args.N_importance > 0 else 4
    img_f_ch = 512
    skips = [4]

    embed_fn_d_pixel, input_ch_d_pixel = get_embedder(args.multires, args.i_embed, 4)
    embeddirs_fn_d_pixel, input_ch_views_d_pixel = get_embedder(args.multires_views, args.i_embed, 3)

    embed_fn_s_pixel, input_ch_s_pixel = get_embedder(args.multires, args.i_embed, 3)
    embeddirs_fn_s_pixel, input_ch_views_s_pixel = get_embedder(args.multires_views, args.i_embed, 3)

    model_d_pixel = PixelNeRF_d(f1D=3, f2D=2, W=256,
                                input_ch=input_ch_d_pixel,
                                input_ch_views=input_ch_views_d_pixel,
                                output_ch=output_ch,
                                img_fea_ch=img_f_ch,
                                use_viewdirsDyn=args.use_viewdirsDyn).to(device)

    device_ids = list(range(torch.cuda.device_count()))
    model_d_pixel = torch.nn.DataParallel(model_d_pixel, device_ids=device_ids)
    grad_vars = list(model_d_pixel.parameters())

    model_s_pixel = PixelNeRF_s(f1D=3, f2D=2, W=256,
                                input_ch=input_ch_s_pixel,
                                input_ch_views=input_ch_views_s_pixel,
                                output_ch=output_ch,
                                img_fea_ch=img_f_ch,
                                use_viewdirs=args.use_viewdirs).to(device)

    model_s_pixel = torch.nn.DataParallel(model_s_pixel, device_ids=device_ids)
    grad_vars += list(model_s_pixel.parameters())

    if args.N_importance > 0:
        raise NotImplementedError

    def network_query_fn_d_pixel(inputs, viewdirs, network_fn, f_projection):
        return run_network(
            inputs, viewdirs, network_fn,
            embed_fn=embed_fn_d_pixel,
            embeddirs_fn=embeddirs_fn_d_pixel,
            netchunk=args.netchunk,
            f_projection=f_projection)

    def network_query_fn_s_pixel(inputs, viewdirs, network_fn, f_projection):
        return run_network(
            inputs, viewdirs, network_fn,
            embed_fn=embed_fn_s_pixel,
            embeddirs_fn=embeddirs_fn_s_pixel,
            netchunk=args.netchunk,
            f_projection=f_projection)

    render_kwargs_train = {
        'network_query_fn_d_pixel': network_query_fn_d_pixel,
        'network_query_fn_s_pixel': network_query_fn_s_pixel,
        'network_fn_d_pixel': model_d_pixel,
        'network_fn_s_pixel': model_s_pixel,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'raw_noise_std': args.raw_noise_std,
        'inference': False,
        'DyNeRF_blending': args.DyNeRF_blending,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else:
        render_kwargs_train['ndc'] = True

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['inference'] = True

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step'] + 1
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # model_d.load_state_dict(ckpt['network_fn_d_state_dict'])
        model_d_pixel.load_state_dict(ckpt['network_fn_d_pixel_state_dict'])
        # model_s.load_state_dict(ckpt['network_fn_s_state_dict'])
        model_s_pixel.load_state_dict(ckpt['network_fn_s_pixel_state_dict'])
        print('Resetting step to', start)

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, encoder


# Ray helpers
def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.
    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * \
         (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * \
         (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def get_grid(H, W, num_img, flows_f, flow_masks_f, flows_b, flow_masks_b):
    # |--------------------|  |--------------------|
    # |       j            |  |       v            |
    # |   i   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')

    grid = np.empty((0, H, W, 8), np.float32)
    for idx in range(num_img):
        grid = np.concatenate((grid, np.stack([i,
                                               j,
                                               flows_f[idx, :, :, 0],
                                               flows_f[idx, :, :, 1],
                                               flow_masks_f[idx, :, :],
                                               flows_b[idx, :, :, 0],
                                               flows_b[idx, :, :, 1],
                                               flow_masks_b[idx, :, :]], -1)[None, ...]))
    return grid


def NDC2world(pts, H, W, f):
    # NDC coordinate to world coordinate
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1., max=1 - 1e-3) - 1)
    pts_x = - pts[..., 0:1] * pts_z * W / 2 / f
    pts_y = - pts[..., 1:2] * pts_z * H / 2 / f
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world


def render_3d_point(H, W, f, pose, weights, pts):
    """Render 3D position along each ray and project it to the image plane.
    """

    c2w = pose
    w2c = c2w[:3, :3].transpose(0, 1)  # same as np.linalg.inv(c2w[:3, :3])

    # Rendered 3D position in NDC coordinate
    pts_map_NDC = torch.sum(weights[..., None] * pts, -2)

    # NDC coordinate to world coordinate
    pts_map_world = NDC2world(pts_map_NDC, H, W, f)

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[:, 3]
    # Rotate
    pts_map_cam = torch.sum(pts_map_world[..., None, :] * w2c[:3, :3], -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat([pts_map_cam[..., 0:1] / (- pts_map_cam[..., 2:]) * f + W * .5,
                           - pts_map_cam[..., 1:2] / (- pts_map_cam[..., 2:]) * f + H * .5],
                          -1)

    return pts_plane


def induce_flow(H, W, focal, pose_neighbor, weights, pts_3d_neighbor, pts_2d):
    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor = render_3d_point(H, W, focal,
                                      pose_neighbor,
                                      weights,
                                      pts_3d_neighbor)
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow


def wapr(K, inv_K, pose, image, depth):
    cam_points = torch.matmul(inv_K[:, :3, :3], image)
    cam_points = depth.view(1, 1, -1) * cam_points
    P = torch.matmul(K, pose)
    cam_points = torch.matmul(P, cam_points)
    return cam_points


def compute_depth_loss(dyn_depth, gt_depth):
    t_d = torch.median(dyn_depth)
    s_d = torch.mean(torch.abs(dyn_depth - t_d))
    dyn_depth_norm = (dyn_depth - t_d) / s_d

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    gt_depth_norm = (gt_depth - t_gt) / s_gt

    return torch.mean((dyn_depth_norm - gt_depth_norm) ** 2)


def normalize_depth(depth):
    return torch.clamp(depth / percentile(depth, 97), 0., 1.)


def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """

    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def save_res(moviebase, ret, fps=None):
    if fps == None:
        if len(ret['rgbs']) < 25:
            fps = 4
        else:
            fps = 24

    for k in ret:
        if 'rgbs' in k:
            # imageio.mimwrite(moviebase + k + '.mp4',
            #                  to8b(ret[k]), fps=fps, quality=8, macro_block_size=1)
            imageio.mimsave(moviebase + k + '.gif',
                             to8b(ret[k]), format='gif', fps=fps)
        elif 'depths' in k:
            # imageio.mimwrite(moviebase + k + '.mp4',
            #                  to8b(ret[k]), fps=fps, quality=8, macro_block_size=1)
            imageio.mimsave(moviebase + k + '.gif',
                             to8b(ret[k]), format='gif', fps=fps)
        elif 'disps' in k:
            # imageio.mimwrite(moviebase + k + '.mp4',
            #                  to8b(ret[k] / np.max(ret[k])), fps=fps, quality=8, macro_block_size=1)
            imageio.mimsave(moviebase + k + '.gif',
                             to8b(ret[k] / np.max(ret[k])), format='gif', fps=fps)
        elif 'sceneflow_' in k:
            # imageio.mimwrite(moviebase + k + '.mp4',
            #                  to8b(norm_sf(ret[k])), fps=fps, quality=8, macro_block_size=1)
            imageio.mimsave(moviebase + k + '.gif',
                             to8b(norm_sf(ret[k])), format='gif', fps=fps)
        elif 'flows' in k:
            # imageio.mimwrite(moviebase + k + '.mp4',
            #                  ret[k], fps=fps, quality=8, macro_block_size=1)
            imageio.mimsave(moviebase + k + '.gif',
                             ret[k], format='gif', fps=fps)
        elif 'dynamicness' in k:
            # imageio.mimwrite(moviebase + k + '.mp4',
            #                  to8b(ret[k]), fps=fps, quality=8, macro_block_size=1)
            imageio.mimsave(moviebase + k + '.gif',
                             to8b(ret[k]), format='gif', fps=fps)
        elif 'disocclusions' in k:
            # imageio.mimwrite(moviebase + k + '.mp4',
            #                  to8b(ret[k][..., 0]), fps=fps, quality=8, macro_block_size=1)
            imageio.mimsave(moviebase + k + '.gif',
                             to8b(ret[k][..., 0]), format='gif', fps=fps)
        elif 'blending' in k:
            blending = ret[k][..., None]
            blending = np.moveaxis(blending, [0, 1, 2, 3], [1, 2, 0, 3])
            # imageio.mimwrite(moviebase + k + '.mp4',
            #                  to8b(blending), fps=fps, quality=8, macro_block_size=1)
            imageio.mimsave(moviebase + k + '.gif',
                             to8b(blending), format='gif', fps=fps)
        elif 'weights' in k:
            # imageio.mimwrite(moviebase + k + '.mp4',
            #                  to8b(ret[k]), fps=fps, quality=8, macro_block_size=1)
            continue
        else:
            raise NotImplementedError


def norm_sf_channel(sf_ch):
    # Make sure zero scene flow is not shifted
    sf_ch[sf_ch >= 0] = sf_ch[sf_ch >= 0] / sf_ch.max() / 2
    sf_ch[sf_ch < 0] = sf_ch[sf_ch < 0] / np.abs(sf_ch.min()) / 2
    sf_ch = sf_ch + 0.5
    return sf_ch


def norm_sf(sf):
    sf = np.concatenate((norm_sf_channel(sf[..., 0:1]),
                         norm_sf_channel(sf[..., 1:2]),
                         norm_sf_channel(sf[..., 2:3])), -1)
    sf = np.moveaxis(sf, [0, 1, 2, 3], [1, 2, 0, 3])
    return sf


# Spatial smoothness (adapted from NSFF)
def compute_sf_smooth_s_loss(pts1, pts2, H, W, f):
    N_samples = pts1.shape[1]

    # NDC coordinate to world coordinate
    pts1_world = NDC2world(pts1[..., :int(N_samples * 0.95), :], H, W, f)
    pts2_world = NDC2world(pts2[..., :int(N_samples * 0.95), :], H, W, f)

    # scene flow in world coordinate
    scene_flow_world = pts1_world - pts2_world

    return L1(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :])


# Temporal smoothness
def compute_sf_smooth_loss(pts, pts_f, pts_b, H, W, f):
    N_samples = pts.shape[1]

    pts_world = NDC2world(pts[..., :int(N_samples * 0.9), :], H, W, f)
    pts_f_world = NDC2world(pts_f[..., :int(N_samples * 0.9), :], H, W, f)
    pts_b_world = NDC2world(pts_b[..., :int(N_samples * 0.9), :], H, W, f)

    # scene flow in world coordinate
    sceneflow_f = pts_f_world - pts_world
    sceneflow_b = pts_b_world - pts_world

    # For a 3D point, its forward and backward sceneflow should be opposite.
    return L2(sceneflow_f + sceneflow_b)
