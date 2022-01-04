import torch
import numpy as np

def score_to_modality(scores: torch.Tensor):
    tensor_list = scores.tolist()
    modality = []
    for row in tensor_list:
        modality.append(row.index(max(row)))
    return modality

class ValTest:
    accuracy = []

    def __init__(self, dl_generator, net, trip_dim, batch_size, device, loss_function, num_modes, datasize):
        self.dl_generator = dl_generator
        self.net = net
        self.trip_dim = trip_dim
        self.batch_size = batch_size
        self.device = device
        self.loss_function = loss_function
        self.num_modes = num_modes
        self.datasize = datasize

    def run(self):

        correct = 0
        total = 0
        val_losses = []
        total_per_mode = [0] * self.num_modes
        correct_per_mode = [0] * self.num_modes
        journeys_eighty_percent_correct = 0

        self.net.eval()  # put net in eval mode

        for data, labels in self.dl_generator():

            self.net.zero_grad()
            lengths = [len(x) for x in data]
            for i, elem in enumerate(data):
               while len(elem) < lengths[0]:
                   elem.append([-1] * self.trip_dim)

            X = np.asarray(data, dtype=np.float)
            input_tensor = torch.from_numpy(X)
            input_tensor = input_tensor.to(self.device)

            output, max_padding_for_this_batch = self.net(input_tensor, lengths)

            for labelz in labels:
               while len(labelz) < max_padding_for_this_batch:
                   labelz.append(-1)

            labels_for_loss = torch.tensor(labels) \
               .view(max_padding_for_this_batch * self.batch_size, -1).squeeze(1).long().to(self.device)

            loss = self.loss_function(output.view(
               max_padding_for_this_batch * self.batch_size, -1),
               labels_for_loss)
            val_losses.append(loss.item())

            for k, journey in enumerate(output):
                journey_correct = 0
                predicted = score_to_modality(journey)

                o = 0
                for o, elem in enumerate(predicted):
                    if labels[k][o] == -1:
                        break
                    total_per_mode[int(labels[k][o])] += 1
                    if labels[k][o] == predicted[o]:
                        correct_per_mode[predicted[o]] += 1
                        correct += 1
                        journey_correct += 1
                    total += 1
                if journey_correct >= (o * 0.80):
                    journeys_eighty_percent_correct += 1

            mode_statistics = []
            for k in range(len(correct_per_mode)):
                if correct_per_mode[k] == 0 or total_per_mode[k] == 0:
                    mode_statistics.append(0)
                    continue
                mode_statistics.append(1 / (total_per_mode[k] / correct_per_mode[k]))

        print('Accuracy: %d %%' % (100 * correct / total))
        print('%% of journeys at least 80%% correct: %d of %d, %d %%' % (
            journeys_eighty_percent_correct, self.datasize, (100 * journeys_eighty_percent_correct / self.datasize)))
        print("Loss: {:.6f}".format(np.mean(val_losses)))
        print("Mode-correct:")
        print(total_per_mode)
        print(mode_statistics)

        self.net.train()
