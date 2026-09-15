import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
import torch.utils.tensorboard as tensorboard

# Load spaCy models
spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")


# Tokenizers
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# Fields
german = Field(
    tokenize=tokenize_ger,
    init_token="<sos>",
    eos_token="<eos>",
    lower=True
)

english = Field(
    tokenize=tokenize_eng,
    init_token="<sos>",
    eos_token="<eos>",
    lower=True
)


# Load dataset
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"),
    fields=(german, english)
)


# Build vocabulary
german.build_vocab(
    train_data,
    min_freq=2,
    max_size=10000
)

english.build_vocab(
    train_data,
    min_freq=2,
    max_size=10000
)


# Encoder
class Encoder(nn.Module):

    def __init__(self, input_size, emb_size, hid_size, n_layers, p):
        super(Encoder, self).__init__()

        self.hidden_size = hid_size

        self.dropout = nn.Dropout(p)

        self.embedding = nn.Embedding(
            input_size,
            emb_size
        )

        self.rnn = nn.LSTM(
            emb_size,
            hid_size,
            n_layers,
            dropout=p
        )

    def forward(self, src):

        embedded = self.dropout(
            self.embedding(src)
        )

        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell


# Decoder
class Decoder(nn.Module):

    def __init__(self, output_size, emb_size, hid_size, n_layers, p):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hid_size

        self.dropout = nn.Dropout(p)

        self.embedding = nn.Embedding(
            output_size,
            emb_size
        )

        self.rnn = nn.LSTM(
            emb_size,
            hid_size,
            n_layers,
            dropout=p
        )

        self.fc_out = nn.Linear(
            hid_size,
            output_size
        )

    def forward(self, input, hidden, cell):

        input = input.unsqueeze(0)

        embedded = self.dropout(
            self.embedding(input)
        )

        output, (hidden, cell) = self.rnn(
            embedded,
            (hidden, cell)
        )

        prediction = self.fc_out(
            output.squeeze(0)
        )

        return prediction, hidden, cell


# Seq2Seq
class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):

        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(
            target_len,
            batch_size,
            target_vocab_size
        ).to(source.device)

        # Encoder reads the source sentence
        hidden, cell = self.encoder(source)

        # Start with <sos>
        x = target[0]

        for t in range(1, target_len):

            # Decoder predicts the next word
            output, hidden, cell = self.decoder(
                x,
                hidden,
                cell
            )

            outputs[t] = output

            # Get the best prediction
            best_guess = output.argmax(1)

            # Teacher forcing
            teacher_force = random.random() < teacher_force_ratio

            if teacher_force:
                x = target[t]
            else:
                x = best_guess

        return outputs


# Training parameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model parameters
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)

encoder_embedding_size = 300
decoder_embedding_size = 300

hidden_size = 1024
num_layers = 2

enc_dropout = 0.5
dec_dropout = 0.5


# Create Encoder and Decoder
encoder_net = Encoder(
    input_size_encoder,
    encoder_embedding_size,
    hidden_size,
    num_layers,
    enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    num_layers,
    dec_dropout
).to(device)


# Create Seq2Seq model
model = Seq2Seq(
    encoder_net,
    decoder_net
).to(device)


# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# Loss function
pad_idx = english.vocab.stoi["<pad>"]

criterion = nn.CrossEntropyLoss(
    ignore_index=pad_idx
)


# TensorBoard
writer = tensorboard.SummaryWriter("runs/loss_plot")

step = 0


# Create iterators
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device
)


# Training
for epoch in range(num_epochs):

    print(f"Epoch {epoch + 1}/{num_epochs}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):

        input_data = batch.src.to(device)
        target_data = batch.trg.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_data,
            target_data
        )

        # Remove <sos>
        outputs = outputs[1:]

        target = target_data[1:]

        # Flatten
        outputs = outputs.reshape(
            -1,
            outputs.shape[2]
        )

        target = target.reshape(-1)

        # Calculate loss
        loss = criterion(
            outputs,
            target
        )

        # Backpropagation
        loss.backward()

        # Prevent very large gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1
        )

        # Update weights
        optimizer.step()

        # TensorBoard
        writer.add_scalar(
            "Training Loss",
            loss.item(),
            step
        )

        step += 1

        if batch_idx % 100 == 0:
            print(
                f"Batch {batch_idx}, Loss: {loss.item():.4f}"
            )


    # Validation
    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in valid_iterator:

            input_data = batch.src.to(device)
            target_data = batch.trg.to(device)

            outputs = model(
                input_data,
                target_data,
                teacher_force_ratio=0
            )

            outputs = outputs[1:]
            target = target_data[1:]

            outputs = outputs.reshape(
                -1,
                outputs.shape[2]
            )

            target = target.reshape(-1)

            loss = criterion(
                outputs,
                target
            )

            total_loss += loss.item()

    valid_loss = total_loss / len(valid_iterator)

    print(
        f"Validation Loss: {valid_loss:.4f}"
    )


# Save model
torch.save(
    model.state_dict(),
    "seq2seq_model.pth"
)

writer.close()

print("Training finished!")