import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # train
    ## files
    parser.add_argument('--train', default='./data/cn_data/train.csv',
                             help="training data")
    parser.add_argument('--eval', default='./data/cn_data/dev.csv',
                             help="evaluation data")

    parser.add_argument('--model_path', default='FImatchE%02dL%.2fA%.2f')
    parser.add_argument('--modeldir', default='./model')
    parser.add_argument('--vec_path', default='./data/vec/snil_trimmed_vec.npy')

    ## vocabulary
    parser.add_argument('--vocab', default='./data/esb.vocab',
                        help="vocabulary file path")
    parser.add_argument('--char_vocab', default='./data/snli.char.vocab',
                        help="vocabulary file path")

    #AE
    parser.add_argument('--ae_layer', default='0,1,2,3,4')

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--preembedding', default=False, type=bool) #本地测试使用
    #parser.add_argument('--preembedding', default=True, type=bool) #实际训练使用
    parser.add_argument('--early_stop', default=20, type=int)
    parser.add_argument('--char_embedding', default=False, type=bool)
    parser.add_argument('--char_dim', default=20, type=int)

    #learning rate 0.0003 is too high
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--num_epochs', default=40, type=int)
    parser.add_argument('--lambda_l2', default=0.004, type=float)

    # model
    # This is also the word embedding size , and must can divide by head num.

    parser.add_argument('--d_model', default=128, type=int,
                        help="hidden dimension of interativate")
    parser.add_argument('--d_ff', default=256, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_transformer', default=3, type=int,
                        help="number of extract blocks")
    parser.add_argument('--blocks', default=3, type=int,
                        help="number of inference_blocks")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen', default=50, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--num_class', default=2, type=int,
                        help="number of class")
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--is_training', default=True, type=bool)


    # test
    parser.add_argument('--test_file', default='./data/snli_test.tsv')
