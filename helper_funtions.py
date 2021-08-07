from torch._C import dtype
from models.imports import *
from imports import *
from tqdm import tqdm


class Help_Funcs(Module):
    def load_data(self):
        data = []
        labels = {}
        labels_idx = -1
        for directory in os.listdir("./data/raw/"):
            labels_idx += 1
            labels["./data/raw/" + directory + "/"] = [labels_idx, -1]
        for label in tqdm(labels.keys()):
            for file in os.listdir(label):
                try:
                    file = label + file
                    labels[label][1] += 1
                    img = Image.open(fr"{file}")
                    image = face_recognition.load_image_file(file)
                    face_locations = face_recognition.face_locations(image)
                    if face_locations != []:
                        face_location = face_locations[0]
                        left = face_location[3] + 5
                        top = face_location[0] + 5
                        right = face_location[1] + 5
                        bottom = face_location[2] + 5
                        img = img.crop((left, top, right, bottom))
                        img.save("./output/0.png")
                        img = cv2.imread("./output/0.png")
                    else:
                        img = cv2.imread(file)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    data.append(
                        [np.array(np.array(img / 255.0)), labels[label][0],]
                    )
                except:
                    print(file)
        np.random.shuffle(data)
        X = []
        y = []
        for d in data:
            X.append(d[0])
            y.append(d[1])
        VAL_SPLIT = 2500
        X_train = np.array(X[:-VAL_SPLIT])
        y_train = np.array(y[:-VAL_SPLIT])
        X_test = np.array(X[-VAL_SPLIT:])
        y_test = np.array(y[-VAL_SPLIT:])
        for key, val in zip(labels.keys(), labels.values()):
            print("*" * 50)
            print(key)
            print(val)
            print("*" * 50)
        return (
            torch.tensor(X_train),
            torch.tensor(y_train),
            torch.tensor(X_test),
            torch.tensor(y_test),
            labels,
        )

    def loss(self, model, criterion, X_batch, y_batch):
        preds = model(X_batch.view(-1, 3, 84, 84).float().to(device))
        loss = criterion(
            preds.view(-1, 1).float().to("cpu").float(),
            y_batch.view(-1, 1).float().to("cpu").float(),
        )
        loss = loss.item()
        return loss

    def accuracy_preds(self, preds, y_batch):
        correct = -1
        total = -1
        for idx in range(len(y_batch)):
            if round(float(preds[idx].float())) == round(float(y_batch[idx].float())):
                correct += 1
            total += 1
        acc = round(correct / total, 3)
        return acc

    def accuracy(self, model, X, y):
        correct = -1
        total = -1
        preds = model(X.to(device).float())
        for idx in range(len(y)):
            y_batch = round(float(y[idx].float()))
            pred = round(float(preds[idx].float()))
            if y_batch == pred:
                correct += 1
            total += 1
        acc = round(correct / total, 3)
        return acc

    def train(self, X_train, y_train, X_test, y_test, model, name):
        try:
            X_test = X_test.view(-1, 3, 84, 84)
            hp = Help_Funcs()
            model = model
            optimizer = config["optimizer"](model.parameters(), lr=config["lr"])
            criterion = config["criterion"]()
            batch_size = config["batch_size"]
            epochs = config["epochs"]
            torch.cuda.empty_cache()
            wandb.init(project=config["PROJECT_NAME"], name=name, sync_tensorboard=True)
            torch.cuda.empty_cache()
            for _ in tqdm(range(epochs)):
                torch.cuda.empty_cache()
                for idx in range(0, len(X_train), batch_size):
                    torch.cuda.empty_cache()
                    X_batch = (
                        X_train[idx : idx + batch_size].to(device).view(-1, 3, 84, 84)
                    )
                    y_batch = y_train[idx : idx + batch_size].to(device)
                    model.to(device)
                    preds = model(X_batch.float())
                    loss = criterion(
                        preds.view(-1, 1).to(device).float(),
                        y_batch.view(-1, 1).float(),
                    )
                    optimizer.zero_grad()
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()
                wandb.log({"v_loss": hp.loss(model, criterion, X_test, y_test)})
                wandb.log({"loss": loss.item()})
                wandb.log({"accuracy": hp.accuracy_preds(preds, y_batch) * 100})
                wandb.log({"val_accuracy": hp.accuracy(model, X_test, y_test) * 100})
            paths = os.listdir("./data/test/")
            new_paths = []
            for path in paths:
                new_paths.append(f"./data/test/{path}")
            hp.get_multiple_preds(paths=new_paths, model=model, IMG_SIZE=84)
            paths = os.listdir("./output/")
            for path in paths:
                wandb.log({f"img/{path}": wandb.Image(cv2.imread(f"./output/{path}"))})
            wandb.finish()
        except:
            torch.cuda.empty_cache()
            wandb.finish()
        return model

    def get_faces(self, paths) -> dict or bool:
        idx = -1
        imgs_dict = {}
        for path in paths:
            image = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) > 0:
                for face_location in tqdm(face_locations):
                    idx += 1
                    im = Image.open(fr"{path}")
                    left = face_location[3] + 5
                    top = face_location[0] + 5
                    right = face_location[1] + 5
                    bottom = face_location[2] + 5
                    im1 = im.crop((left, top, right, bottom))  # Croping into the Image
                    im1.save(f"./output/{idx}.png")
                    # The below is the proccess of adding the idx and idx img to the dir and adding it to 'imgs_dict'
                    if path in list(imgs_dict.keys()):
                        imgs_dict[path][0].append(idx)
                        imgs_dict[path][1].append(f"./output/{idx}.png")
                    else:
                        imgs_dict[path] = [
                            [idx],
                            [f"./output/{idx}.png"],
                        ]
        return imgs_dict

    def get_multiple_preds(
        self,
        paths,
        model,
        labels={1: "female", 0: "male", "1": "female", "0": "male"},
        num_of_times=1,
        IMG_SIZE=84,
    ) -> dict:
        with torch.no_grad():
            preds = {}
            hp = Help_Funcs()
            faces_results = hp.get_faces(paths)
            for _ in range(num_of_times):
                imgs = []
                for key, val in zip(faces_results.keys(), faces_results.values()):
                    img = cv2.imread(val[1][0])
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    imgs.append(img)
                if imgs == []:
                    break
                preds_model = model(
                    torch.tensor(np.array(imgs))
                    .view(-1, 3, 84, 84)
                    .to(config["device"])
                    .float()
                )
                for key, val, pred in zip(
                    faces_results.keys(), faces_results.values(), preds_model
                ):
                    pred = int(round(float(pred)))
                    try:
                        preds[val[0][0]][0][int(pred)] += 1
                    except Exception as e:
                        preds[val[0][0]] = [{0: 0, 1: 0}, [key, val[1][0]]]
                        preds[val[0][0]][0][pred] += 1
            results = {}
            for idx, log in zip(preds.keys(), preds.values()):
                files = log[1]
                log = log[0]
                best_class = -1
                if log[0] < log[1]:
                    best_class = 1
                elif log[0] > log[1]:
                    best_class = 0
                img = cv2.imread(files[1])
                results[idx] = [
                    [best_class, files[0], files[1], img.tolist()],
                    files[0],
                    files[1],
                ]
                plt.figure(figsize=(10, 7))
                plt.imshow(img)
                plt.title(f"{labels[best_class]}")
                plt.savefig(f"./output/{idx}.png")
            return results
