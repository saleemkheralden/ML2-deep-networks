from train_model_q2 import *

# Sample from a category and starting letter
def evaluate_model_q2(category, start_letter='A', max_length=20):
    import __main__
    setattr(__main__, "RNN", RNN)
    rnn = pickle.load(open('model_q2.pkl', 'rb'))

    # same error here as the CNN evaluate file, Invalid magic number
    # rnn = RNN(n_letters, 128, n_letters)
    # rnn.load_state_dict(torch.load('model_q2.pkl', map_location=lambda storage, loc: storage))

    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name