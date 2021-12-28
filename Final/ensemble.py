import pandas as pd
import os

clip = 0.90
classes = 8

def clip_csv(csv_file, clip, classes):
    # Read the submission file
    df = pd.read_csv(csv_file, index_col=0)

    # Clip the values
    df = df.clip(lower=(1.0 - clip)/float(classes - 1), upper=clip)

    # Normalize the values to 1
    df = df.div(df.sum(axis=1), axis=0)

    # Save the new clipped values
    df.to_csv('final_submission.csv')
    print(df.head(10))


if __name__ == '__main__':
    df1 = pd.read_csv('resnet50_epoch10_crop.csv')
    df2 = pd.read_csv('regnet_x_8gf_crop.csv')
    ensemble = df1.copy()
    columns = list(ensemble.columns)
    print(columns)
    ensemble[columns[1:]] = (df1[columns[1:]] + df2[columns[1:]]) / 2
    ensemble.to_csv('ensemble.csv', index=False)
    clip_csv('ensemble.csv', clip, classes)
