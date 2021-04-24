print('import main file..')
import copy
import sys
import os
sys.path.append('../')
from models import *
from data_reader import *
from transformers import AlbertModel, AdamW, AlbertTokenizer
from sklearn.metrics import precision_recall_fscore_support


def make_model(args):
    print('Making model...', file=SHELL_OUT_FILE, flush=True)
    c = copy.deepcopy
    bert_model = AlbertModel.from_pretrained('albert-base-v2')
    model = albert(bert_model, 5)
    model.init_parameters()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.CrossEntropyLoss()
    print('Done', file=SHELL_OUT_FILE, flush=True)
    return model, optimizer, criterion


def evaluate(criterion, output, label, outputs, labels):
    with torch.no_grad():
        loss = criterion(output.transpose(-1, -2), label)
        outputs.extend(np.concatenate(np.argmax(output.cpu().numpy(), axis=-1).tolist()).tolist())
        labels.extend(np.concatenate(label.cpu().numpy().astype(np.int).tolist()).tolist())
    return loss.item()


def run(args):
    args.use_cuda
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.shell_print == 'file':
        SHELL_OUT_FILE = open(args.output_dir + 'shell_out', 'a+', encoding='utf-8')
    else:
        SHELL_OUT_FILE = sys.stdout
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    """calling the reader from reader file"""
    processor = jade_processor(AlbertTokenizer.from_pretrained('albert-base-v2'), args.max_seq_length)
    reader = jade_reader(args.batch_size)

    # Create model
    model, optimizer, criterion = make_model(args)
    # Load model if it exists
    file_name = args.output_dir + 'model_' + args.suffix
    start_epoch = 0
    if os.path.exists(file_name):
        state = torch.load(file_name)
        model.load_state_dict(state['state_dict'])
        start_epoch = state['epoch']
    print(type(model), file=SHELL_OUT_FILE, flush=True)
    if args.multi_gpu:
        model = nn.DataParallel(model)
    print(type(model), file=SHELL_OUT_FILE, flush=True)
    model = model.cuda() if excute.USE_CUDA else model
    model_dict = {'last': model, 'acc': None, 'recall': None, 'fval': None, 'loss': None}
    epoch = {'last': 0, 'acc': 0, 'recall': 0, 'fval': 0, 'loss': 0}
    best_acc = 0
    best_recall = 0
    best_fval = 0
    best_loss = 1e9

    if args.do_train:
        train_examples = reader.get_train_examples(args.data_dir)
        total_train_examples = len(train_examples)
        for ep in range(args.epoch):
            print("######## Training ########", file=SHELL_OUT_FILE, flush=True)
            print('Epoch:', start_epoch + ep, file=SHELL_OUT_FILE, flush=True)
            loss_train = []
            model.train()
            print("\rTrain Step: {}/{} Loss: {}".format(0, total_train_examples, 0), file=SHELL_OUT_FILE, flush=True)
            for i, example in enumerate(train_examples):
                inputs, labels = processor.convert_examples_to_tensor(example)
                prediction = model(*inputs)
                loss = criterion(prediction.transpose(-1, -2), labels)
                loss_train.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print("\rTrain Step: {}/{} Loss: {}".format(i + 1, total_train_examples, loss), file=SHELL_OUT_FILE,
                          flush=True)

            if args.do_eval:
                print("\n######## Evaluating ########", file=SHELL_OUT_FILE, flush=True)
                eval_examples = reader.get_dev_examples(args.data_dir)
                output_eval = []
                label_eval = []
                loss_eval = []
                loss_eval_all = 0
                model.eval()
                total_eval_examples = len(eval_examples)
                with torch.no_grad():
                    print("\rEval Step: {}/{}".format(0, total_eval_examples), end='\r', file=SHELL_OUT_FILE,
                          flush=True)
                    for i, example in enumerate(eval_examples):
                        if (i + 1) % 100 == 0:
                            print("\rEval Step: {}/{}".format(i + 1, total_eval_examples), end='\r',
                                  file=SHELL_OUT_FILE, flush=True)
                        inputs, labels = processor.convert_examples_to_tensor(example)
                        prediction = model(*inputs)
                        loss = evaluate(criterion, prediction, labels, output_eval, label_eval)
                        loss_eval_all += loss

                    print("\rEval Step: {}/{}".format(total_eval_examples, total_eval_examples), file=SHELL_OUT_FILE,
                          flush=True)
                    loss_eval_all /= total_eval_examples
                    loss_eval.append(loss_eval_all)
                    print('Loss:', loss_eval_all, file=SHELL_OUT_FILE, flush=True)

                    acc, recall, fval, _ = precision_recall_fscore_support(label_eval, output_eval, average='weighted')
                    print("Accuracy:", acc, file=SHELL_OUT_FILE, flush=True)
                    print("Recall:", recall, file=SHELL_OUT_FILE, flush=True)
                    print("F-score", fval, file=SHELL_OUT_FILE, flush=True)

                    save_model = copy.deepcopy(model)
                    save_model = save_model.module.cpu() if args.multi_gpu else save_model.cpu()
                    epoch['last'] = start_epoch + ep
                    file_name = args.output_dir + '/model_last'
                    state = {'epoch': epoch['last'] + 1, 'state_dict': save_model.state_dict()}
                    torch.save(state, file_name)
                    # save model with best accuracy on dev set
                    if acc > best_acc:  # Useless
                        best_acc = acc
                        epoch['acc'] = start_epoch + ep
                        model_dict['acc'] = save_model
                        file_name = args.output_dir + 'model_acc'
                        state = {'epoch': epoch['acc'] + 1, 'state_dict': model_dict['acc'].state_dict()}
                        torch.save(state, file_name)
                        # torch.save(model, file_name)
                    # save model with best recall@1 on dev set
                    if recall > best_recall:
                        best_recall = recall
                        epoch['recall'] = start_epoch + ep
                        model_dict['recall'] = save_model
                        file_name = args.output_dir + 'model_recall'
                        state = {'epoch': epoch['recall'] + 1, 'state_dict': model_dict['recall'].state_dict()}
                        torch.save(state, file_name)
                    # save model with best recall@1 on dev set
                    if fval > best_fval:
                        best_fval = fval
                        epoch['fval'] = start_epoch + ep
                        model_dict['fval'] = save_model
                        file_name = args.output_dir + 'model_fval'
                        state = {'epoch': epoch['fval'] + 1, 'state_dict': model_dict['fval'].state_dict()}
                        torch.save(state, file_name)
                    # save model with best loss on dev set
                    if loss_eval_all < best_loss:
                        best_loss = loss_eval_all
                        epoch['loss'] = start_epoch + ep
                        model_dict['loss'] = save_model
                        file_name = args.output_dir + 'model_loss'
                        state = {'epoch': epoch['loss'] + 1, 'state_dict': model_dict['loss'].state_dict()}
                        torch.save(state, file_name)

    if args.do_predict:
        print("####################### Testing ##########################################", file=SHELL_OUT_FILE,
              flush=True)
        test_examples = reader.get_test_examples(args.data_dir)
        for suffix in ['last', 'acc', 'recall', 'fval', 'loss']:
            # for suffix in ['last']:
            # Create model
            model, optimizer, criterion = make_model(args)
            # Load model if it exists
            file_name = args.output_dir + 'model_' + suffix

            if os.path.exists(file_name):
                state = torch.load(file_name)
                model.load_state_dict(state['state_dict'])
            else:
                continue
            print(type(model), file=SHELL_OUT_FILE, flush=True)
            if args.multi_gpu:
                model = nn.DataParallel(model)
            print(type(model), file=SHELL_OUT_FILE, flush=True)
            model = model.cuda() if excute.USE_CUDA else model

            loss_test = 0
            tokens_out = []
            masks_out = []
            output_out = []
            output_test = []
            label_test = []
            back_label = []

            model.eval()
            total_test_examples = len(test_examples)
            with torch.no_grad():
                for i, example in enumerate(test_examples):
                    if (i + 1) % 100 == 0:
                        print("\rTrain Step: {}/{}".format(i + 1, total_test_examples), end='\r', file=SHELL_OUT_FILE,
                              flush=True)
                    inputs, labels = processor.convert_examples_to_tensor(example)
                    predictions = model(*inputs)
                    back_label.extend(np.concatenate(np.argmax(predictions.cpu().numpy(), axis=-1).tolist()).tolist())
                    tokens_out.extend(processor.convert_tensor_to_tokens(inputs[0][:, 1: -1]))
                    masks_out.extend(inputs[1][:, 1: -1].cpu().numpy().tolist())
                    output_out.extend(predictions.cpu().numpy().tolist())
                    loss = evaluate(criterion, predictions, labels, output_test, label_test)
                    loss_test += loss

                print("\n#### " + suffix.upper() + " ####", file=SHELL_OUT_FILE, flush=True)
                loss_test /= total_test_examples
                print("Loss:", loss_test, file=SHELL_OUT_FILE, flush=True)

                output_out = np.argmax(output_out, axis=-1).tolist()
                output_result = []
                for seq, mask, sep in zip(tokens_out, masks_out, output_out):
                    single_seq = []
                    string = ''
                    for c, m, o in zip(seq, mask, sep):
                        if m == 0:
                            break;
                        if c[:2] != '##':
                            if c[0] != '[' and c[-1] != ']':
                                string += ' ' + c
                        else:
                            string += c[2:]
                        if o == 0:
                            single_seq.append(string.strip())
                            single_seq.append("NA")
                            string = ''
                        if o == 1:
                            single_seq.append(string.strip())
                            single_seq.append("S")
                            string = ''
                        if o == 2:
                            single_seq.append(string.strip())
                            single_seq.append("D")
                            string = ''
                        if o == 3:
                            single_seq.append(string.strip())
                            single_seq.append("A")
                            string = ''
                    if string != '':
                        single_seq.append(string.strip())
                    output_result.append(single_seq)

                output_file_name = args.output_dir + 'result_' + suffix + '.txt'
                with open(output_file_name, 'w', encoding='utf-8') as f:
                    for ip in output_result:
                        f.write(str(ip))
                        f.write('\n')

                    print('Write Done')

                acc, recall, fval, _ = precision_recall_fscore_support(label_test, output_test, average='micro')
                print("Accuracy:", acc, file=SHELL_OUT_FILE, flush=True)
                print("Recall:", recall, file=SHELL_OUT_FILE, flush=True)
                print("F-score", fval, file=SHELL_OUT_FILE, flush=True)


if __name__ == '__main__':
    run(args)
