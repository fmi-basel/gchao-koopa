# cuddly-octo-enigma

* Change directory to `cd luigi`
* Change config file
* Activate conda environment
* Start luigi deamon: `luigid --background --logdir ../log_luigi`
* Start actual process: `nice -n19 python -m luigi --module merge Merge --workers ??` (2-4)
* Monitor task on `localhost:8087`
