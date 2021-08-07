from torchvision.models import *
from helper_funtions import *
from models.imports import *
from imports import *
from models.cnn import CNN


hp = Help_Funcs()
X_train, y_train, X_test, y_test, labels = hp.load_data()

class TL_Model(Module):
    def __init__(self,model,num_of_classes=1):
        super().__init__()
        self.model = model
        self.output = Linear(1000,num_of_classes)
    
    def forward(self,X):
        preds = self.model(X)
        preds = self.output(preds)
        preds = Sigmoid()(preds)
        return preds
model = TL_Model(shufflenet_v2_x1_0(pretrained=True))
model = hp.train(X_train, y_train, X_test, y_test, model, f"Final-0",)
torch.save(model, "./trained_models/model-g.pt")
torch.save(model, "./trained_models/model-g.pth")
torch.save(model.state_dict(), "./trained_models/model-g-sd.pt")
torch.save(model.state_dict(), "./trained_models/model-g-sd.pth")
