import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_reconstruction(obs, mu_rec, var_t, mt, mm, ml):
    obs = np.flipud(obs.squeeze().detach().cpu().numpy())
    mu_rec = np.flipud(mu_rec.squeeze().detach().cpu().numpy())
    var_t = np.flipud(var_t.squeeze().detach().cpu().numpy())
    mt = np.flipud(mt.squeeze().detach().cpu().numpy())
    mm = np.flipud(mm.squeeze().detach().cpu().numpy())
    ml = np.flipud(ml.squeeze().detach().cpu().numpy())

    bg_cmap = mcolors.ListedColormap(["#e5e4e2", "white"])
    settings = {"interpolation": "none", "aspect": "auto"}

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    plt.setp(axes, xticks=[], yticks=[])

    for ax in axes:
        ax.imshow(ml, cmap=bg_cmap, **settings)

    m_all = mt * ml
    m_vis = m_all * mm
    m_hid = m_all * (1 - mm)

    m_obs = np.ma.masked_where(m_all == 0, obs)

    vmin = m_obs.min()
    vmax = m_obs.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    se = ((obs - mu_rec) * m_all) ** 2
    rse = np.sqrt(se)
    rse_score = np.sqrt(se.sum() / m_all.sum())

    titles = [
        r'$\mathbf{x}_t \odot \mathbf{m}_m$',
        r'$\mathbf{x}_t$',
        r'$\hat{\mathbf{x}}_t$',
        r'$\boldsymbol{\sigma}$',
        r'$\text{RMSE}_\text{all}$=' + f'{rse_score:.3f}',
    ]

    data = [
        np.ma.masked_where(m_vis == 0, obs),
        np.ma.masked_where(m_all == 0, obs),
        np.ma.masked_where(ml == 0, mu_rec),
        np.ma.masked_where(ml == 0, np.sqrt(var_t)),
        np.ma.masked_where(m_all == 0, rse)
    ]

    cmaps = [
        'viridis',
        'viridis',
        'viridis',
        'coolwarm',
        'coolwarm',
    ]

    norms = [
        norm,
        norm,
        norm,
        plt.Normalize(vmin=0, vmax=1.5),
        plt.Normalize(vmin=0, vmax=1.5),
    ]

    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_ylabel('DIRECT', labelpad=10, size='large')
        im = ax.imshow(data[i], norm=norms[i], cmap=cmaps[i], **settings)
        ax.set_title(titles[i], fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.1)

    plt.tight_layout()
    plt.show()