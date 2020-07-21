import torch
import torchvision
from torch.utils import data
from torchvision import transforms, datasets
import torch
import torchvision
from tqdm import trange
import torch.optim as optim

batch_size = 4
data_size = 2000
step_count = (int) (data_size / batch_size)
epoch_count = 20

transform = transforms.Compose([
        transforms.ToTensor()
])

train_set = datasets.CocoDetection(root='/home/data/train2017',
                                   annFile='/home/data/annotations/instances_train2017.json',
                                   transform=transform)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.cuda()

a = train_set[0][0]
b = train_set[0][1]

images = [ a.cuda() for i in range(batch_size) ]
boxes = torch.stack([ torch.FloatTensor(b1['bbox']) for b1 in b ], dim=0)
labels = torch.cat([ torch.LongTensor([ b1['category_id'] ]) for b1 in b ], dim=-1)
targets = [ {'boxes':boxes.cuda(), 'labels':labels.cuda()} for i in range(batch_size) ]




"""
for b1 in b:
    boxes.append(torch.cat(b1['bbox'], dim=-1).cuda())
    ca_id.append(b1['category_id'])
    c.append( {'boxes':torch.stack(boxes, dim=0).float(), 'labels':torch.cat(ca_id, dim=-1)} )
#c = [ {'boxes':torch.stack(boxes, dim=0).float(), 'labels':torch.cat(ca_id, dim=-1)} ]
"""


optimizer = optim.SGD(model.parameters(), lr=0.001)
model.train()

for epoch in trange(epoch_count):
    for step in trange(step_count):
        loss = model(images, targets)
        losses = sum(loss for loss in loss.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

print("finish !")

