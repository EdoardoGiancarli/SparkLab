import torch
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

    pred_img, pred_pars = model(x_img, x_pars, t, c_img, c_pars)

    return


if __name__ == '__main__':

    shape: tuple[int, int, int] = (10, 320, int(2.5 * 120))
    model_factory = {
        'dim': 8,
        # 'use_convnext': False,
    }

    main(shape, model_factory)