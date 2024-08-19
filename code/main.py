if __name__ == "__main__":

    from tracker import Tracker

    folder_name = 'potato'
    scale_down = 3
    skip_images = 5
    suppress = 0.1
    conf_prob = 0.4
    img_h = 692
    img_w = 1029
    device = 'mps'

    """
    Usually no need to change. 
    """
    debug_js = False
    stream_from_mem = True
    dp_threshold = 16

    tracker = Tracker(
        name=folder_name,
        scale_down=scale_down,
        skip_images=skip_images,
        suppress=suppress,
        conf_prob=conf_prob,
        img_h=img_h,
        img_w=img_w,
        debug_js=debug_js,
        device=device,
        stream_from_mem=stream_from_mem,
        dp_threshold=dp_threshold
    )
    tracker.preprocess()
    tracker.predict()
    tracker.plot_areas(show=True)
