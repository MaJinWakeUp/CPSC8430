import torch
from dataset import MyDataset, Text2Embedding
from net import Seq2Seq
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    batch_size = 8
    num_workers = 4
    caption_length = 50
    video_length = 80
    num_vfeatures = 4096

    data_dir = './MLDS_hw2_1_data'
    embed = Text2Embedding('word_dict.json', max_length=caption_length)
    train_dataset = MyDataset(data_dir, 'training', pre_extracted=True, caption_transform=embed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    num_words = embed.num_words
    bos_idx = embed.word2id['<BOS>']
    eos_idx = embed.word2id['<EOS>']
    model = Seq2Seq(num_words, frame_dim=num_vfeatures, hidden=256, dropout=0.2, v_step=video_length, c_step=caption_length, bos_idx=bos_idx)
    model = model.cuda()
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 10

    losses_train = []
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        epoch_loss_train = 0
        for i, (vfeat, caption) in progress_bar:
            vfeat = vfeat.cuda()
            caption = caption.cuda()
            num_captions = caption.size(1)
            iter_loss = 0
            for j in range(num_captions):
                cur_caption = caption[:, j, :].cuda()
                # print(cur_caption)
                output, prob = model(vfeat)
                prob = prob.view(-1, prob.shape[-1])
                target = cur_caption[:, 1:].contiguous().view(-1)
                # print(prob.shape, target.shape)
                loss = criterion(prob, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_loss += loss.item()
            iter_loss = iter_loss / num_captions
            progress_bar.set_description(f'Epoch {epoch} batch {i} loss: {iter_loss}')
            epoch_loss_train += iter_loss
        epoch_loss_train /= len(train_loader)
        print(f'Epoch {epoch} loss: {epoch_loss_train}')
        losses_train.append(epoch_loss_train)

    torch.save(model.state_dict(), f'./final_model.pt')

    # Plot loss curve
    fig = plt.figure()
    plt.plot(losses_train, label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('loss_curve.png')
    plt.close()
    

if __name__ == '__main__':
    main()