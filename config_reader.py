import copy
import multiprocessing as mp
import os
import re
import time
import random
import json
import pynvml
import numpy as np
import torch.version

pynvml.nvmlInit()


def process_configs(target, arg_parser):
    args, _ = arg_parser.parse_known_args()
    ctx = mp.get_context('fork')

    subprocess = []
    if "ALL_GPU" in os.environ:
        all_gpu_queue = list(map(int, os.environ["ALL_GPU"].split(",")))
    else:
        all_gpu_queue = [0, 1, 2, 3, 4, 5, 6, 7]
    gpu_queue = []
    waittime = 240
    gpu_just_used = []
    for run_args, _run_config, _run_repeat in _yield_configs(arg_parser, args):
        if "eval" in run_args.label:
            waittime = 60
        if run_args.seed == -1:
            run_args.seed = random.randint(0, 1000)
        # debug
        if run_args.debug:
            target(run_args)
        while not run_args.cpu and (len(gpu_queue) == 0 or len(gpu_queue) < run_args.world_size):
            gpu_queue = []
            candidate_gpu = list(set(all_gpu_queue) - set(gpu_just_used))
            # print(candidate_gpu)
            for index in candidate_gpu:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if run_args.ignore_gpu_usage:
                        gpu_queue.append(index)
                    elif meminfo.used / 1024 / 1024 < 1500:
                        gpu_queue.append(index)
                except Exception as e:
                    pass
            print(gpu_queue)
            if len(gpu_queue) < run_args.world_size:
                print(
                    f"Need {run_args.world_size} GPUs for DDP Training, but only {len(gpu_queue)} free devices: {gpu_queue}. Waiting for Free GPU ......")
                time.sleep(waittime)
                gpu_just_used = []
            elif len(gpu_queue) == 0:
                print("Need 1 GPU for Normal Training, All are busy. Waiting for Free GPU ......")
                time.sleep(waittime)
                gpu_just_used = []
            else:
                print("Avaliable devices: ", gpu_queue)
            # gpu_just_used = []
        # CPU Training:
        if run_args.cpu:
            print("########### Using CPU Training ###########")
            print("Using Random Seed", run_args.seed)
            p = ctx.Process(target=target, args=(run_args,))
            subprocess.append(p)
            p.start()
            time.sleep(1)
        # GPU Training
        else:
            # GPU DDP Training
            if run_args.world_size != -1:
                print("########### Using GPU DDP Training ###########")
                print("Using devices: ", gpu_queue)
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_queue))
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))

                for local_rank in range(run_args.world_size):
                    gpu_just_used.append(gpu_queue[0])
                    gpu_queue = gpu_queue[1:]
                    run_args.local_rank = local_rank
                    print("Using Random Seed", run_args.seed)
                    p = ctx.Process(target=target, args=(run_args,))
                    subprocess.append(p)
                    p.start()
                time.sleep(1)
            # GPU Normal Training
            if run_args.world_size == -1:
                print("########### Using GPU Normal Training ###########")
                if run_args.device_id in gpu_queue:
                    pass
                else:
                    device_id = random.sample(gpu_queue, k=1)[0]
                    run_args.device_id = device_id
                gpu_just_used.append(run_args.device_id)
                gpu_queue.remove(run_args.device_id)
                print("Using devices: ", run_args.device_id)
                print("Using Random Seed", run_args.seed)
                p = ctx.Process(target=target, args=(run_args,))
                subprocess.append(p)
                p.start()
                time.sleep(1)

    list(map(lambda x: x.join(), subprocess))


def _read_config(path):
    lines = open(path).readlines()

    runs = []
    run = [1, dict()]
    for line in lines:
        stripped_line = line.strip()

        # continue in case of comment
        if stripped_line.startswith('#'):
            continue

        if not stripped_line:
            if run[1]:
                runs.append(run)

            run = [1, dict()]
            continue

        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            repeat = int(stripped_line[1:-1])
            run[0] = repeat
        else:
            key, value = stripped_line.split('=')
            key, value = (key.strip(), value.strip())
            run[1][key] = value

    if run[1]:
        runs.append(run)

    return runs


def _convert_config(config):
    config_list = []
    for k, v in config.items():
        if v == "None":
            continue
        if v.startswith("["):
            v = v[1:-1].replace(",", "")
        if v.lower() == 'true':
            config_list.append('--' + k)
        elif v.lower() != 'false':
            config_list.extend(['--' + k] + v.split(' '))
    return config_list


def _yield_configs(arg_parser, args, verbose=True):
    _print = (lambda x: print(x)) if verbose else lambda x: x

    if args.config:
        config = _read_config(args.config)

        for run_repeat, run_config in config:
            print("-" * 50)
            print("Config:")
            # print(run_config)

            args_copy = copy.deepcopy(args)
            run_config = copy.deepcopy(run_config)
            config_list = _convert_config(run_config)
            run_args = arg_parser.parse_args(config_list, namespace=args_copy)

            run_args_list = []
            # batch eval
            if run_args.label == "batch_eval_flag":
                save_path = run_args.model_path
                # save_model_type = run_args.save_model_type
                for dirpath, dirnames, filenames in sorted(os.walk(save_path), key=lambda x: x[0]):
                    if dirpath.endswith("_model"):
                        print(dirpath)
                        dataset_name = re.match(".*/(.*?)_train", dirpath).group(1)
                        # print(dirpath)
                        # exp_label=dirpath.split("/")[-3]
                        # exp_time=dirpath.split("/")[-2]
                        args_path = "/".join(dirpath.split("/")[:-1]) + "/args.json"
                        args_dict = json.load(open(args_path))

                        run_args.label = dataset_name + "_eval"
                        if "train_dev" in args_dict["train_path"]:
                            run_args.dataset_path = args_dict["train_path"].replace("train_dev", "test")
                        else:
                            run_args.dataset_path = args_dict["train_path"].replace("train", "test")
                        run_args.model_path = dirpath
                        run_args.tokenizer_path = dirpath
                        run_args.types_path = args_dict["types_path"]
                        run_args.log_path = args_dict["log_path"]

                        run_args.seed = args_dict["seed"]
                        run_args.model_type = args_dict["model_type"]
                        run_args.weight_decay = args_dict["weight_decay"]
                        # run_args.no_overlapping =args_dict["no_overlapping"]
                        # run_args.no_partial_overlapping =args_dict["no_partial_overlapping"]
                        # run_args.no_duplicate =args_dict["no_duplicate"]

                        if run_args.eval_batch_size == -1:
                            run_args.eval_batch_size = args_dict["eval_batch_size"]

                        run_args.prop_drop = args_dict["prop_drop"]

                        run_args.pos_size = args_dict["pos_size"]
                        run_args.char_lstm_layers = args_dict["char_lstm_layers"]
                        run_args.char_lstm_drop = args_dict["char_lstm_drop"]
                        run_args.char_size = args_dict["char_size"]
                        run_args.use_pos = args_dict["use_pos"]
                        run_args.use_glove = args_dict["use_glove"]
                        run_args.use_char_lstm = args_dict["use_char_lstm"]
                        run_args.pool_type = args_dict["pool_type"]
                        run_args.wordvec_path = args_dict["wordvec_path"]

                        if "use_aux_loss" in args_dict:
                            run_args.use_aux_loss = args_dict["use_aux_loss"]
                        else:
                            run_args.use_aux_loss = True

                        if "use_lstm" in args_dict:
                            run_args.use_lstm = args_dict["use_aux_loss"]
                        else:
                            run_args.use_lstm = True

                        run_args.nil_weight = args_dict["nil_weight"]

                        run_args.match_boundary_weight = args_dict["match_boundary_weight"]
                        run_args.match_class_weight = args_dict["match_class_weight"]
                        run_args.loss_boundary_weight = args_dict["loss_boundary_weight"]
                        run_args.loss_class_weight = args_dict["loss_class_weight"]
                        run_args.match_solver = args_dict["match_solver"]

                        run_args.entity_queries_num = args_dict["entity_queries_num"]
                        run_args.num_dec_layer = args_dict["num_dec_layer"]
                        run_args.split_epoch = args_dict["split_epoch"]
                        run_args.epochs = args_dict["epochs"]

                        if run_args.cls_threshold == -1 and run_args.boundary_threshold != -1:
                            for cls_threshold in np.arange(0, 1, 0.1):
                                run_args_instance = copy.deepcopy(run_args)
                                run_args_instance.cls_threshold = cls_threshold
                                run_args_list.append(run_args_instance)

                        if run_args.boundary_threshold == -1 and run_args.cls_threshold != -1:
                            for boundary_threshold in np.arange(0, 1, 0.1):
                                run_args_instance = copy.deepcopy(run_args)
                                run_args_instance.boundary_threshold = boundary_threshold
                                run_args_list.append(run_args_instance)

                        if run_args.cls_threshold == -1 and run_args.boundary_threshold == -1:
                            for cls_threshold in np.arange(0, 1, 0.1):
                                for boundary_threshold in np.arange(0, 1, 0.1):
                                    run_args_instance = copy.deepcopy(run_args)
                                    run_args_instance.cls_threshold = cls_threshold
                                    run_args_instance.boundary_threshold = boundary_threshold
                                    run_args_list.append(run_args_instance)

                        if run_args.cls_threshold != -1 and run_args.boundary_threshold != -1:
                            run_args_list.append(copy.deepcopy(run_args))
            else:
                run_args_list.append(run_args)

            for run_args in run_args_list:
                print(run_args)
                print("Repeat %s times" % run_repeat)
                print("-" * 50)
                for iteration in range(run_repeat):
                    _print("Iteration %s" % iteration)
                    _print("-" * 50)

                    yield copy.deepcopy(run_args), run_config, run_repeat

            # time.sleep(3)

    else:
        yield args, None, None
