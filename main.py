from utils.config import parse_args
from utils.data_loader import get_data_loader

from models.sslgan_gp import SSLGAN_GP
from models.sslgan_sn import SSLGAN_SN

def main(args):
    model = None
    if args.model == 'SSLGAN_GP':
        model = SSLGAN_GP(args)
    elif args.model == 'SSLGAN_SN':
        model = SSLGAN_SN(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)


if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)