from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense, Concatenate, GlobalMaxPooling1D, Dropout, Reshape, Flatten, Bidirectional
from tensorflow.keras.models import Model
import tensorflow as tf

# Model Building Function
def build_model(num_sender_ips, sequence_length, num_ips):
    input_source_ip_freq = Input(shape=(1, num_sender_ips), name='source_ip_freq')
    input_time_between_emails = Input(shape=(sequence_length, 1), name='time_between_emails')
    input_timestamp = Input(shape=(sequence_length, 1), name='timestamp')
    input_time_discrepancies = Input(shape=(sequence_length, 1), name='time_discrepancies')
    input_x_originating_ip = Input(shape=(num_ips,), name='x_originating_ip')

    lstm_input = Concatenate(axis=-1)([input_time_between_emails, input_timestamp, input_time_discrepancies])
    lstm_output_temporal = Bidirectional(LSTM(128, return_sequences=True))(lstm_input)
    lstm_output_temporal = GlobalMaxPooling1D()(lstm_output_temporal)

    x_originating_ip_expanded = Reshape((num_ips, 1))(input_x_originating_ip)
    x_originating_ip_expanded_flat = Flatten()(x_originating_ip_expanded)
    source_ip_freq_flat = Flatten()(input_source_ip_freq)

    cnn_input = Concatenate()([x_originating_ip_expanded_flat, source_ip_freq_flat])
    cnn_input_reshaped = Reshape((cnn_input.shape[1], 1))(cnn_input)
    cnn_output = Conv1D(128, 3, activation='relu')(cnn_input_reshaped)
    cnn_output = GlobalMaxPooling1D()(cnn_output)

    combined_features = Concatenate()([lstm_output_temporal, cnn_output])
    x = Dense(256, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_sender_ips, activation='softmax')(x)

    model = Model(inputs=[input_source_ip_freq, input_time_between_emails, input_timestamp, input_time_discrepancies, input_x_originating_ip], outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
