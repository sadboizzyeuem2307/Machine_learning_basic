from lib import *

# path
path_train_img = r'D:\Machine_learning_basic\Penguins vs Turtles\train\train'
path_valid_img = r'D:/Machine_learning_basic/Penguins vs Turtles/valid/valid'
path_train_json = r'D:\Machine_learning_basic\Penguins vs Turtles\train_annotations'
path_valid_json = r'D:\Machine_learning_basic\Penguins vs Turtles\valid_annotations'

json_train = pd.read_json(path_train_json)
json_valid = pd.read_json(path_valid_json)

classes_train = list(json_train['category_id'].unique())
print(classes_train)

json_train =json_train.drop(['id', 'bbox', 'area', 'segmentation', 'iscrowd'], axis =1)

json_train['category_id']= json_train['category_id'].replace({1:'Penguin',2:'Turtle'})
print (json_train.head())

classes_train = list(json_train['category_id'].unique())
print (classes_train)

json_train.columns=['File-paths', 'Labels']
print(json_train.head())

train_image_dir= path_train_img
train_imglist= sorted(os.listdir(train_image_dir))
train_paths=[]
for i in range (len(train_imglist)):
    train_paths.append(os.path.join(train_image_dir, train_imglist[i]))    
print (train_paths[0])

json_train['File-paths'] = train_paths
print (json_train.head())

valid_json = pd.read_json(path_valid_json)
valid_json= valid_json.drop(['id', 'bbox', 'area', 'segmentation', 'iscrowd'], axis =1)

valid_json['category_id']= valid_json['category_id'].replace({1:'Penguin',2:'Turtle'})
valid_json.columns=['File-paths', 'Labels']

valid_image_dir = path_valid_img
valid_imglist = sorted(os.listdir(valid_image_dir))
valid_paths= []

for i in range (len(valid_imglist)):
    valid_paths.append(os.path.join(valid_image_dir, valid_imglist[i]))
valid_json['File-paths']= valid_paths
print (valid_json)

img_size=(224,224) # image size

def resize_image(image_path, new_size):
    image = cv.imread(image_path)
    resized_image = cv.resize(image, new_size)
    cv.imwrite(image_path, resized_image)

for image_path in json_train['File-paths']:
    resize_image(image_path, img_size)

for image_path in valid_json['File-paths']:
    resize_image(image_path, img_size)

file_label_mapping = dict(zip(json_train['File-paths'], json_train['Labels']))

for filename in os.listdir(path_train_img):
    
    old_path = os.path.join(path_train_img, filename)
    new_label = file_label_mapping.get(old_path)
    new_filename = f'{new_label}_{filename}' if new_label else filename
    new_path = os.path.join(path_train_img, new_filename)
    os.rename(old_path, new_path)


file_label_mapping = dict(zip(valid_json['File-paths'], valid_json['Labels']))

for filename in os.listdir(path_valid_img):
    
    old_path = os.path.join(path_valid_img, filename)
    new_label = file_label_mapping.get(old_path)
    new_filename = f'{new_label}_{filename}' if new_label else filename
    new_path = os.path.join(path_valid_img, new_filename)
    os.rename(old_path, new_path)

for filename in os.listdir(path_train_img):
    old_path = os.path.join(path_train_img, filename)
    
    
    base_name, extension = os.path.splitext(filename)
    
    if '_' in base_name:
        
        prefix, name = base_name.split('_', 1)
    else:
        
        prefix = base_name
        name = ''
    
    
    new_filename = f'{prefix}{extension}'
    
    new_path = os.path.join(path_train_img, new_filename)
    
    count = 1
    while os.path.exists(new_path):
        
        new_filename = f'{prefix}_{count}{extension}'
        new_path = os.path.join(path_train_img, new_filename)
        count += 1
    
    os.rename(old_path, new_path)

for filename in os.listdir(path_valid_img):
    old_path = os.path.join(path_valid_img, filename)
    
    
    base_name, extension = os.path.splitext(filename)
    
    if '_' in base_name:
        
        prefix, name = base_name.split('_', 1)
    else:
        
        prefix = base_name
        name = ''
    
    
    new_filename = f'{prefix}{extension}'
    
    new_path = os.path.join(path_valid_img, new_filename)
    
    count = 1
    while os.path.exists(new_path):
        
        new_filename = f'{prefix}_{count}{extension}'
        new_path = os.path.join(path_valid_img, new_filename)
        count += 1
    
    os.rename(old_path, new_path)

turtle_dir = os.path.join(path_train_img, 'turtle')
penguin_dir = os.path.join(path_train_img, 'penguin')


os.makedirs(turtle_dir, exist_ok=True)
os.makedirs(penguin_dir, exist_ok=True)


for filename in os.listdir(path_train_img):
    if filename.endswith('.jpg'):
        label = filename.split('_')[0]  
        if label.lower() == 'turtle':
            
            src_path = os.path.join(path_train_img, filename)
            dst_path = os.path.join(turtle_dir, filename)
            shutil.move(src_path, dst_path)
        elif label.lower() == 'penguin':
            # Di chuyển vào thư mục penguin
            src_path = os.path.join(path_train_img, filename)
            dst_path = os.path.join(penguin_dir, filename)
            shutil.move(src_path, dst_path)

turtle_dir = os.path.join(path_valid_img, 'turtle')
penguin_dir = os.path.join(path_valid_img, 'penguin')


os.makedirs(turtle_dir, exist_ok=True)
os.makedirs(penguin_dir, exist_ok=True)


for filename in os.listdir(path_valid_img):
    if filename.endswith('.jpg'):
        label = filename.split('_')[0]  
        if label.lower() == 'turtle':
            
            src_path = os.path.join(path_valid_img, filename)
            dst_path = os.path.join(turtle_dir, filename)
            shutil.move(src_path, dst_path)
        elif label.lower() == 'penguin':
            
            src_path = os.path.join(path_valid_img, filename)
            dst_path = os.path.join(penguin_dir, filename)
            shutil.move(src_path, dst_path)