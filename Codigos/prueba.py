def run_solver(total, proc_id, result, fouts, locks):
    with open(fouts[proc_id], 'a') as openfile:
        for i in range(10):
            with locks[proc_id]:
                openfile.write('hi\n')
                openfile.flush()


if __name__ == '__main__':
    processes = []
    with Manager() as manager:
        fouts = manager.list(['0.txt', '1.txt', '2.txt', '3.txt'])
        locks = manager.list([Lock() for fout in fouts])

        for proc_id in range(os.cpu_count()):
            processes.append(Process(
                target=run_solver, args=(
                    int(total/os.cpu_count()), proc_id, result, fouts, locks
                )
            ))