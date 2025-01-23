# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Mitten can be used as a tool to create simple benchmarks of workloads in existing frameworks, with simple edits from
native code. As an example, let's use a tutorial from PyTorch:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Let's assume we've already gone through the tutorial and want to change it into a Mitten pipeline.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from nvmitten.pipeline import BenchmarkMetric, ScratchSpace, Operation, Pipeline


"""The first step we need to define is downloading, generating, and/or preprocessing our dataset. In the case of this
example, the downloading and preprocessing can be done with a single step with torchvision.datasets and transforms.

Note: We could do this as a separate, prior step, but Mitten pipeline are meant to be standalone executables that
execute all the required steps to run a workload. Let's wrap the steps in a Mitten Operation."""
class CIFAR10DownloadOp(Operation):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def run(self, scratch_space, dependency_outputs):
        # From the tutorial. Since initializing the dataset does the download itself, it is the only part that needs to
        # be in .run(). Note that we don't actually need to store the object, we're just using the download
        # functionality.
        # The next tutorial shows how to write a download operation for GET-able web objects, which is how Mitten is
        # intended to be used.
        dataroot = scratch_space.working_dir(namespace="data")
        trainset = torchvision.datasets.CIFAR10(root=dataroot,
                                                train=True,
                                                download=True,
                                                transform=CIFAR10DownloadOp.transform)
        testset = torchvision.datasets.CIFAR10(root=dataroot,
                                               train=False,
                                               download=True,
                                               transform=CIFAR10DownloadOp.transform)
        return {"trainset": trainset,
                "testset": testset}

    @classmethod
    def immediate_dependencies(cls):
        # No operations need to be run before this, so just return None, or an empty Set.
        return None


# Copy the network architecture code directly from the tutorial.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""Now to implement our benchmark, we need to decide what metrics we want to measure. For the training process, these
could be:
    1. How many iterations it takes to convergence.
    2. How much overall time it takes to train.
    3. The time it takes per batch, or samples / second throughput.

Option (1) could be implemented with a running average loss, but the tutorial simply trains for 2 epochs. For the sake
of simplicity, let's implement (3).
"""
class TrainingOp(Operation):

    def __init__(self, batch_size=4, num_workers=2, lr=1e-3, momentum=0.9, device=None):
        """Operations should only accept keyword arguments in __init__. The default values here are taken from the
        PyTorch tutorial.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.momentum = momentum
        self.device = device

    @classmethod
    def immediate_dependencies(cls):
        return {CIFAR10DownloadOp}

    def run(self, scratch_space, dependency_outputs):
        # Grab trainset from dependency_outputs and construct the DataLoaders.
        trainloader = torch.utils.data.DataLoader(dependency_outputs[CIFAR10DownloadOp]["trainset"],
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.num_workers)
        net = Net().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum)

        elapsed, n_samples = 0, 0
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                n_samples += len(inputs)

                # Time the relevant component
                t_start = time.time()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                t_end = time.time()
                elapsed += t_end - t_start

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        # Example on how to save the trained model into the ScratchSpace
        modelroot = scratch_space.working_dir(namespace="model")
        torch.save(net.state_dict(), modelroot / "cifar_net.pth")

        qps = n_samples / elapsed
        return {"qps": (qps, BenchmarkMetric("samples / second")),
                "network": net}


"""Once our model is trained, we can define an operation to benchmark its accuracy as well.
"""
class InferenceOp(Operation):

    def __init__(self, batch_size=4, num_workers=2, device=None):
        """Operations should only accept keyword arguments in __init__. The default values here are taken from the
        PyTorch tutorial.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    @classmethod
    def immediate_dependencies(cls):
        return {CIFAR10DownloadOp, TrainingOp}

    def run(self, scratch_space, dependency_outputs):
        testloader = torch.utils.data.DataLoader(dependency_outputs[CIFAR10DownloadOp]["testset"],
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.num_workers)
        net = dependency_outputs[TrainingOp]["network"]
        elapsed = 0

        # From the tutorial:
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running images through the network
                time_start = time.time()
                outputs = net(images)
                time_end = time.time()
                elapsed += time_end - time_start
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return {"accuracy": (accuracy, BenchmarkMetric("%")),
                "qps": (total / elapsed, BenchmarkMetric("samples / second"))}


"""Since this pipeline has multiple metrics as output, we have to define an "aggregation" operation to return all the
metrics as the pipeline's output. Mitten pipelines do not support multiple outputs, so this is the canonical way to do
this.
"""
class AggregateOp(Operation):

    @classmethod
    def immediate_dependencies(cls):
        return {TrainingOp, InferenceOp}

    def run(self, scratch_space, dependency_outputs):
        return {"training_qps": dependency_outputs[TrainingOp]["qps"],
                "inference_qps": dependency_outputs[InferenceOp]["qps"],
                "accuracy": dependency_outputs[InferenceOp]["accuracy"]}


# Now to run everything! Let's declare our pipeline.
operations = [AggregateOp,
              CIFAR10DownloadOp,
              InferenceOp,
              TrainingOp]

# Note `operations` is defined out of execution order simply to demonstrate that Mitten Pipelines will automatically
# determine execution order at runtime based on dependencies.
# We can also define configs for each op like so:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
bs = 4
config = {TrainingOp: {"batch_size": bs,
                       "device": device},
          InferenceOp: {"batch_size": bs,
                        "device": device}}
scratch_space = ScratchSpace("/tmp/mitten/1_PyTorch")

pipeline = Pipeline(scratch_space, operations, config)
metrics = pipeline.run()

for k, v in metrics.items():
    print(f"{k}: {v}")
