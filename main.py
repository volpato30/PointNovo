import os
import torch
import subprocess
import cProfile
import logging
import logging.config
from config import config
from train_func import train, build_model, validation, perplexity
from data_reader import DeepNovoDenovoDataset, collate_func, DeepNovoTrainDataset, DBSearchDataset, denovo_collate_func
from db_searcher import DataBaseSearcher
from psm_ranker import PSMRank
from model import InferenceModelWrapper, device, SpectrumEncoding
from denovo import IonCNNDenovo
import time
from writer import DenovoWriter, PercolatorWriter
import deepnovo_worker_test
from deepnovo_dia_script_select import find_score_cutoff

logger = logging.getLogger(__name__)


def main():
    logger.info(f"use_lstm: {config.use_lstm}")
    if config.FLAGS.train:
        logger.info("training mode")
        train()
    elif config.FLAGS.search_denovo:
        logger.info("denovo mode")
        data_reader = DeepNovoDenovoDataset(feature_filename=config.denovo_input_feature_file,
                                            spectrum_filename=config.denovo_input_spectrum_file)
        denovo_data_loader = torch.utils.data.DataLoader(dataset=data_reader, batch_size=config.batch_size,
                                                         shuffle=False,
                                                         num_workers=config.num_workers,
                                                         collate_fn=denovo_collate_func)
        denovo_worker = IonCNNDenovo(config.MZ_MAX,
                                     config.knapsack_file,
                                     beam_size=config.FLAGS.beam_size)
        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
        model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, init_net)
        writer = DenovoWriter(config.denovo_output_file)
        start_time = time.time()
        with torch.no_grad():
            denovo_worker.search_denovo(model_wrapper, denovo_data_loader, writer)
            # cProfile.runctx("denovo_worker.search_denovo(model_wrapper, denovo_data_loader, writer)", globals(), locals())
        logger.info(f"de novo {len(data_reader)} spectra takes {time.time() - start_time} seconds")
    elif config.FLAGS.valid:
        valid_set = DeepNovoTrainDataset(config.input_feature_file_valid,
                                         config.input_spectrum_file_valid)
        valid_data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                        batch_size=config.batch_size,
                                                        shuffle=False,
                                                        num_workers=config.num_workers,
                                                        collate_fn=collate_func)
        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
        forward_deepnovo.eval()
        backward_deepnovo.eval()
        validation_loss = validation(forward_deepnovo, backward_deepnovo, init_net, valid_data_loader)
        logger.info(f"validation perplexity: {perplexity(validation_loss)}")

    elif config.FLAGS.test:
        logger.info("test mode")
        worker_test = deepnovo_worker_test.WorkerTest()
        worker_test.test_accuracy()

        # show 95 accuracy score threshold
        accuracy_cutoff = 0.95
        accuracy_file = config.accuracy_file
        score_cutoff = find_score_cutoff(accuracy_file, accuracy_cutoff)

    elif config.FLAGS.search_db:
        logger.info("data base search mode")
        start_time = time.time()
        db_searcher = DataBaseSearcher(config.db_fasta_file)
        dataset = DBSearchDataset(config.search_db_input_feature_file,
                                  config.search_db_input_spectrum_file,
                                  db_searcher)
        num_spectra = len(dataset)

        def simple_collate_func(train_data_list):
            return train_data_list

        data_reader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=config.num_db_searcher_worker,
                                                  collate_fn=simple_collate_func)

        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
        forward_deepnovo.eval()
        backward_deepnovo.eval()

        writer = PercolatorWriter(config.db_output_file)
        psm_ranker = PSMRank(data_reader, forward_deepnovo, backward_deepnovo, writer, num_spectra)
        psm_ranker.search()
        writer.close()
        # call percolator
        with open(f"{config.db_output_file}" + '.psms', "w") as fw:
            subprocess.run(["percolator", "-X", "/tmp/pout.xml", f"{config.db_output_file}"],
                           stdout=fw)

    elif config.FLAGS.serialize_model:
        device = torch.device("cpu")
        logger.info("serialize the trained model into a distributable format")
        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)

        # forward_deepnovo = forward_deepnovo.to(device)
        # backward_deepnovo = backward_deepnovo.to(device)
        #
        # # create fake inputs
        # with torch.no_grad():
        #     fake_input_ones = (torch.ones((1, 1, config.vocab_size, config.num_ion)).float().to(device),
        #                        torch.ones((1, config.MAX_NUM_PEAK)).float().to(device),
        #                        torch.ones((1, config.MAX_NUM_PEAK)).float().to(device),
        #                        )
        #     forward_output = forward_deepnovo(*fake_input_ones).cpu().numpy().flatten()
        #     forward_script_model = torch.jit.trace(forward_deepnovo, fake_input_ones)
        #     backward_output = backward_deepnovo(*fake_input_ones).cpu().numpy().flatten()
        #     backward_script_model = torch.jit.trace(backward_deepnovo, fake_input_ones)
        #     logger.info(f"forward output:\n{forward_output}")
        #     logger.info(f"backward output:\n{backward_output}")

        # if not os.path.exists(os.path.join(config.train_dir, "dist")):
        #     os.mkdir(os.path.join(config.train_dir, "dist"))
        # forward_script_model.save(os.path.join(config.train_dir, "dist", "forward_scripted.pt"))
        # backward_script_model.save(os.path.join(config.train_dir, "dist", "backward_scripted.pt"))

        forward_deepnovo = forward_deepnovo.to(device)
        backward_deepnovo = backward_deepnovo.to(device)
        if config.use_lstm:
            init_net = init_net.to(device)

        # create fake inputs
        with torch.no_grad():
            if config.use_lstm:
                fake_input_ones = (torch.ones((1, 1, config.vocab_size, config.num_ion)).float().to(device),
                                   torch.ones((1, config.MAX_NUM_PEAK)).float().to(device),
                                   torch.ones((1, config.MAX_NUM_PEAK)).float().to(device),
                                   torch.ones((1, 1)).long().to(device),
                                   (torch.ones((1, 1, config.embedding_size)).float().to(device),
                                    torch.ones((1, 1, config.embedding_size)).float().to(device)),
                                   )
                # fake_input_spectrum = (
                #     torch.ones((1, config.MAX_NUM_PEAK)).float().to(device),
                #     torch.ones((1, config.MAX_NUM_PEAK)).float().to(device),
                # )
                # fake_input = {"forward": fake_input_ones}
                # forward_script_model = torch.jit.trace_module(forward_deepnovo, fake_input)
                # backward_script_model = torch.jit.trace_module(backward_deepnovo, fake_input)
                forward_script_model = torch.jit.trace(forward_deepnovo, fake_input_ones)
                backward_script_model = torch.jit.trace(backward_deepnovo, fake_input_ones)
                forward_output = forward_deepnovo(*fake_input_ones)[0].cpu().numpy().flatten()
                backward_output = backward_deepnovo(*fake_input_ones)[0].cpu().numpy().flatten()
                # _ = forward_deepnovo.get_spectrum_representation(*fake_input_spectrum)
                # forward_script_model = torch.jit.trace(forward_deepnovo, fake_input_ones)

            else:
                fake_input_ones = (torch.ones((1, 1, config.vocab_size, config.num_ion)).float().to(device),
                                   torch.ones((1, config.MAX_NUM_PEAK)).float().to(device),
                                   torch.ones((1, config.MAX_NUM_PEAK)).float().to(device),
                                   )
                forward_output = forward_deepnovo(*fake_input_ones).cpu().numpy().flatten()
                forward_script_model = torch.jit.trace(forward_deepnovo, fake_input_ones)
                backward_output = backward_deepnovo(*fake_input_ones).cpu().numpy().flatten()
                backward_script_model = torch.jit.trace(backward_deepnovo, fake_input_ones)
            logger.debug(f"forward output:\n{forward_output}")
            logger.debug(f"backward output:\n{backward_output}")

        forward_script_model.save(config.forward_model_path)
        backward_script_model.save(config.backward_model_path)
        if config.use_lstm:
            init_script_model = torch.jit.script(init_net)
            init_script_model.save(config.initnet_model_path)
            spectrum_encoding = SpectrumEncoding(d_model=config.embedding_size, max_len=config.n_position,
                                                 spectrum_reso=config.spectrum_reso)
            script_encoding = torch.jit.script(spectrum_encoding)
            script_encoding.save(config.spectrum_encoding_model_path)
    elif config.FLAGS.onnx:
        device = torch.device("cpu")
        logger.info("serialize the trained model into ONNX format")
        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
        forward_deepnovo = forward_deepnovo.to(device)
        backward_deepnovo = backward_deepnovo.to(device)
        if config.use_lstm:
            init_net = init_net.to(device)

        # create fake inputs
        with torch.no_grad():
            if config.use_lstm:
                raise ValueError()

            else:
                fake_input_ones = (torch.ones((1, 1, config.vocab_size, config.num_ion)).float().to(device),
                                   torch.ones((1, config.MAX_NUM_PEAK)).float().to(device),
                                   torch.ones((1, config.MAX_NUM_PEAK)).float().to(device),
                                   )
                input_names = ["theoretical_ion_mz", "spectrum_mz", "spectrum_intensity"]
                output_names = ["aa_prob_logits"]
                torch.onnx.export(forward_deepnovo, fake_input_ones, "onnx/forward_model.onnx", verbose=True,
                                  input_names=input_names,
                                  output_names=output_names)
                torch.onnx.export(backward_deepnovo, fake_input_ones, "onnx/backward_model.onnx", verbose=True,
                                  input_names=input_names,
                                  output_names=output_names)
    else:
        raise RuntimeError("unspecified mode")


if __name__ == '__main__':
    log_file_name = 'PointNovo.log'
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)
    main()
