from lib import *

def data(path):
    label = {}
    data = {}
    count = 0
    for i in os.listdir(path):
        file = path + '/' + i
        label[i] = count
        lst = []
        for j in os.listdir(file):
            img_path = file + '/' + j
            img = cv.imread(img_path)
            img = cv.resize(img, (64, 64))
            lst.append(img)
        data[i] = lst
        count += 1
    return data, label


def visualiztion(data):
    plt.figure(figsize=(20, 10))
    lst_name = [i for i in data]
    fx = random.randint(0, len(lst_name) - 1)
    for i in range(16):
        th = random.randint(0, len(data[lst_name[fx]]) - 1)
        plt.subplot(4, 4, i + 1)
        plt.imshow(cv.cvtColor(data[lst_name[fx]][i], cv.COLOR_BGR2RGB))
        plt.title(lst_name[fx])
        plt.axis('off')

def hog_img(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img, (3, 3), 0.01)
    fg, img = hog(img_gray, orientations= 12, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    img = np.uint8(img)
    return img

def hog_data(data):
    data_ = {}
    for i in data:
        lst = []
        for j in data[i]:
            j = hog_img(j)
            lst.append(j)
        data_[i] = lst
    return data_

def data_label(data, label):
    data_model = []
    label_model = []
    for i in data:
        for j in data[i]:
            j = (j - j.mean()) / j.std()
            j = j.flatten()
            data_model.append(j)
            label_model.append(label[i])
    data_model = np.array(data_model)
    label_model = np.array(label_model)
    data_model = np.concatenate((data_model, label_model[:, None]), axis=1)
    data_model = shuffle(data_model)
    return data_model

def train_test_split_(data):
    y = data[:, -1]
    X = np.delete(data, -1, 1)
    return X, y