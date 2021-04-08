def record_GPU_info():
    jobid_gpuid = args.outfile.split('-')[-1]
    tmp_split = jobid_gpuid.split('_')[-1]
    if len(tmp_split) == 2:
        gpuid = tmp_split[-1]
    else:
        gpuid = 0
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
    gpuname = nvidia_smi.nvmlDeviceGetName(handle)
    gpuname = gpuname.decode("utf-8")
    log_dict['gpuname'] = gpuname
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    gpu_mem_usage = str(_bytes_to_megabytes(info.used)) + '/' + str(_bytes_to_megabytes(info.total)) + ' MB'
    log_dict['gpu_mem_usage'] = gpu_mem_usage
    gpu_util = 0
    for i in range(5):
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        gpu_util += res.gpu
        time.sleep(1)
    gpu_util = gpu_util / 5
    log_dict['gpu_util'] = str(gpu_util) + '%'
    
    return None


def fill_running(df: pd.DataFrame):
    for index, row in df.iterrows():
        if 'State' not in list(row.index) or row['State'] in [None, np.nan, 'RUNNING']:
            try:
                jobid = row['outfile'].split('-')[-1].split('_')[0]  # extract job id from outfile name
                seff = os.popen('seff ' + jobid)  # get job information
                for line in seff.readlines():
                    line = line.split(
                        ': ')  # must have space to be differentiated from time format 00:12:34
                    if len(line) == 2:
                        key, value = line
                        key = '_'.join(key.split(' '))  # change 'CPU utilized' to 'CPU_utilized'
                        value = value.split('\n')[0]
                        df.at[index, key] = value
            except:
                pass
    return df

def correct_type(df: pd.DataFrame):
    for column in df:
        ori_type = type(df[column].to_list()[-1])
        if ori_type is int:
            df[column] = df[column].astype('Int64')  # correct type
    return df


def record_experiment(record_file: str, current_id: Optional[int] = None):
    if current_id is None:  # before the experiment
        lock = FileLock(record_file + ".lock")
        with lock:  # with this lock,  open a file for exclusive access
            with open(record_file, 'a') as csv_file:
                if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
                    new_id = 1
                    df = pd.DataFrame()
                else:
                    df = pd.read_csv(record_file)
                    last_id = df['ID'].to_list()[-1]
                    new_id = int(last_id) + 1
                mypath = Path(new_id, check_id_dir=True)  # to check if id_dir already exist

                date = datetime.date.today().strftime("%Y-%m-%d")
                time = datetime.datetime.now().time().strftime("%H:%M:%S")
                # row = [new_id, date, time, ]
                idatime = {'ID': new_id, 'start_date': date, 'start_time': time}

                args_dict = vars(args)
                idatime.update(args_dict)
                if len(df) == 0:
                    df = pd.DataFrame([idatime])  # need a [] , or need to assign the index for df
                else:
                    for key, value in idatime.items():
                        df.at[new_id - 1, key] = value  #

                df = fill_running(df)
                df = correct_type(df)

                df.to_csv(record_file, index=False)
                df.to_csv(mypath.id_dir + '/' + record_file, index=False)
                shutil.copy(record_file, 'cp_' + record_file)
        return new_id
    else:  # at the end of this experiments, find the line of this id, and record the final information
        lock = FileLock(record_file + ".lock")
        with lock:  # with this lock,  open a file for exclusive access
            df = pd.read_csv(record_file)
            index = df.index[df['ID'] == current_id].to_list()
            if len(index) > 1:
                raise Exception("over 1 row has the same id", id)
            elif len(index) == 0:  # only one line,
                index = 0
            else:
                index = index[0]

            date = datetime.date.today().strftime("%Y-%m-%d")
            time = datetime.datetime.now().time().strftime("%H:%M:%S")
            df.at[index, 'end_date'] = date
            df.at[index, 'end_time'] = time

            # usage
            f = "%Y-%m-%d %H:%M:%S"
            t1 = datetime.datetime.strptime(df['start_date'][index] + ' ' + df['start_time'][index], f)
            t2 = datetime.datetime.strptime(df['end_date'][index] + ' ' + df['end_time'][index], f)
            elapsed_time = check_time_difference(t1, t2)
            df.at[index, 'elapsed_time'] = elapsed_time

            mypath = Path(current_id)  # evaluate old model
            for mode in ['train', 'valid', 'test']:
                lock2 = FileLock(mypath.loss(mode) + ".lock")
                # when evaluating old mode3ls, those files would be copied to new the folder
                with lock2:
                    loss_df = pd.read_csv(mypath.loss(mode))
                    best_index = loss_df['mae_end5'].idxmin()
                    log_dict['metrics_min'] = 'mae_end5'
                    loss = loss_df['loss'][best_index]
                    mae = loss_df['mae'][best_index]
                    mae_end5 = loss_df['mae_end5'][best_index]
                df.at[index, mode + '_loss'] = round(loss, 2)
                df.at[index, mode + '_mae'] = round(mae, 2)
                df.at[index, mode + '_mae_end5'] = round(mae_end5, 2)

            for key, value in log_dict.items():  # write all log_dict to csv file
                if type(value) is np.ndarray:
                    str_v = ''
                    for v in value:
                        str_v += str(v)
                        str_v += '_'
                    value = str_v
                df.loc[index, key] = value
                if type(value) is int:
                    df[key] = df[key].astype('Int64')

            for column in df:
                if type(df[column].to_list()[-1]) is int:
                    df[column] = df[column].astype('Int64')  # correct type

            args_dict = vars(args)
            args_dict.update({'ID': current_id})
            for column in df:
                if column in args_dict.keys() and type(args_dict[column]) is int:
                    df[column] = df[column].astype(float).astype('Int64')  # correct str to float and then int

            df.to_csv(record_file, index=False)
            df.to_csv(mypath.id_dir + '/' + record_file, index=False)
            shutil.copy(record_file, 'cp_' + record_file)

