def plot_map():
    import colorsys
    import folium
    from mmm import GeoUtils as GU
    from tqdm import tqdm

    base_setting = {
        'x_range': (-175, -64),
        'y_range': (18, 71),
        'fineness': (20, 20),
        'numdata_sqrt_oneclass': 32
    }
    datas, (mean, std) = GU.base_dataset(**base_setting)
    num_class = base_setting['fineness'][0] * base_setting['fineness'][1]

    _map = folium.Map(
        location=[40.0, -100.0],
        zoom_start=4,
        tiles='Stamen Terrain'
    )

    HSV_tuples = [(x * 1.0 / num_class, 1.0, 1.0) for x in range(num_class)]
    RGB_tuples = [
        '#%02x%02x%02x' % (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255))
        for x in list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    ]

    for item in tqdm(datas):
        label, locate = item['labels'], item['locate']
        locate = [locate[1], locate[0]]
        radius = 150
        label = label[0]

        folium.Circle(
            radius=radius,
            location=locate,
            popup=label,
            color=RGB_tuples[label],
            fill=False,
        ).add_to(_map)

    return _map


def visualize_classmap(weight='../datas/geo_base/outputs/learned/200weight.pth',
                       lat_range=(25, 50), lng_range=(-60, -125), unit=0.5,
                       limited=None):
    import colorsys
    import folium
    import numpy as np
    import torch
    from mmm import GeoBaseNet
    from mmm import GeoUtils as GU

    base_setting = {
        'x_range': (-175, -64),
        'y_range': (18, 71),
        'fineness': (20, 20),
        'numdata_sqrt_oneclass': 32
    }
    _, (mean, std) = GU.base_dataset(**base_setting)
    num_class = base_setting['fineness'][0] * base_setting['fineness'][1]

    BR_settings = {
        'x_range': (-175, -64),
        'y_range': (18, 71),
        'fineness': (20, 20),
        'mean': mean,
        'std': std
    }

    model = GeoBaseNet(
        class_num=num_class,
        momentum=0.9,
        network_setting=BR_settings,
    )
    model.loadmodel(weight)

    # -------------------------------------------------------------------------
    # make points
    lat_range, lng_range = sorted(lat_range), sorted(lng_range)
    lats = np.arange(lat_range[0], lat_range[1], unit)
    lngs = np.arange(lng_range[0], lng_range[1], unit)

    # -------------------------------------------------------------------------
    # make base map
    _map = folium.Map(
        location=[40.0, -100.0],
        zoom_start=4,
        tiles='Stamen Terrain'
    )

    # make colors list
    limited = [i for i in range(num_class)] if limited is None else limited
    limited = set([i for i in limited if 0 <= i < num_class])
    convert_idx = {}
    cnt = 0
    for idx in range(num_class):
        if idx in limited:
            convert_idx[idx] = cnt
            cnt += 1

    color_num = len(convert_idx)
    HSV_tuples = [(x * 1.0 / color_num, 1.0, 1.0) for x in range(color_num)]
    RGB_tuples = [
        '#%02x%02x%02x' % (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255))
        for x in list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    ]

    # -------------------------------------------------------------------------
    # plot
    for lat in lats:
        for lng in lngs:
            label = model.predict(torch.Tensor([[lng, lat]]), labeling=True)
            label = int(label[0])
            if label not in limited:
                continue

            folium.Circle(
                radius=150,
                location=[lat, lng],
                popup=label,
                color=RGB_tuples[convert_idx[label]],
                fill=False
            ).add_to(_map)

    return _map


if __name__ == "__main__":
    # plot_map()
    # visualize_classmap()

    print('finish.')
