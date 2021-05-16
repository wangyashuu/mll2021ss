def extract_raw_features_and_targets(files, classname = ""):
    X, Y = [], []
    for name in files:
        if isinstance(files[name], str):
            X.append(files[name])
            Y.append(classname)
        else:
            X_sub, Y_sub = extract_raw_features_and_targets(files[name], classname + name)
            X = X + X_sub
            Y = Y + Y_sub
    return X, Y
