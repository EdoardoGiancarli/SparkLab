import torch
from torch.amp import GradScaler, autocast
from pkdev.model import Unet

def main(shape, model_factory: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {device}...\n')
    to_device = lambda x: x.to(device)

    b, n, m = shape

    t = torch.ones(b, dtype=torch.long)
    x_img = torch.ones((b, 1, n, m))
    x_pars = torch.ones((b, 3))
    c_img = torch.ones((b, 1, 70, 20))
    c_pars = torch.ones((b, 3))

    model = Unet(**model_factory)

    t = to_device(t)
    x_img, x_pars = map(to_device, (x_img, x_pars))
    c_img, c_pars = map(to_device, (c_img, c_pars))
    model = to_device(model)

    # pred_img, pred_pars = model(x_img, x_pars, t, c_img, c_pars)                             # NOTE: model isworking CHECK

    # assert all(map(lambda x, y: x.shape == y.shape, (x_img, x_pars), (pred_img, pred_pars))) # NOTE: shape test CHECK

    # NOTE: memory tests (32, 320, 120)
    loss_fn = lambda x, y: torch.square(x - y).sum() / (n * m)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(device)
    model = torch.compile(model, mode="max-autotune")

    # optimiser.zero_grad()
    # with autocast(device_type=device):
    #     print('Model/loss computation...\n')
    #     pred_img, pred_pars = model(x_img, x_pars, t, c_img, c_pars)
    #     lval = sum(map(loss_fn, (x_img, x_pars), (pred_img, pred_pars)))

    # print('Optim Step...\n')
    # scaler.scale(lval).backward()
    # scaler.step(optimiser)
    # scaler.update()

    explanation = torch._dynamo.explain(model)(x_img, x_pars, t, c_img, c_pars)
    print(explanation)

    # NOTE:
    #   * x_img shape must be multiple of depth_lvl (for 3 level, (304, 1264) is ok --> crop 4 pxs on H)
    #   * very comp-heavy
    #   * Legion laptop max memoryload: (32, 320, 120)  -using only Attention (with `sdpa`, but each epoch takes maybe 10min)
    #   * test `torch.compile()`
    #       - first test with base U-Net and (32, 320, 120): optim step was done, GPU memory very low (~800MB), took less than 10min
    #       - seems that compiling extend the epoch duration of minutes!

    return


if __name__ == '__main__':

    shape: tuple[int, int, int] = (1, 304, 120)
    model_factory = {
        'dim': 8,
        # 'use_convnext': False,
    }

    main(shape, model_factory)