from models.cnn_lstm import train_cnn_lstm

def train_cnn_lstm_model(frames, seq_len=20, batch_size=2, epochs=5, save_path="models/cnn_lstm.h5"):
    return train_cnn_lstm(frames, seq_len=seq_len, batch_size=batch_size, epochs=epochs, save_path=save_path)
