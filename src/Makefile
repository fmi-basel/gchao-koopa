clean:
	rm -rf /tungstenfs/scratch/gchao/eichbast/tests/fish_2d_analysis/*
	rm -rf /tungstenfs/scratch/gchao/eichbast/tests/fish_3d_analysis/*
	rm -rf /tungstenfs/scratch/gchao/eichbast/tests/live_analysis/*

live:
	cp /tungstenfs/scratch/gchao/eichbast/cuddly-octo-enigma/src/live.cfg /tungstenfs/scratch/gchao/eichbast/cuddly-octo-enigma/src/luigi.cfg
	nice -n10\
		python -m luigi\
		--module colocalize Colocalize\
		--FileID "20210303_smGCN4_HighGFPpool_50ms_1000gain_488-30_561-50_CTR_2"\
		--ChannelPairIndex 0
	ls -lh /tungstenfs/scratch/gchao/eichbast/tests/live_analysis/*

two:
	cp /tungstenfs/scratch/gchao/eichbast/cuddly-octo-enigma/src/fish_2d.cfg /tungstenfs/scratch/gchao/eichbast/cuddly-octo-enigma/src/luigi.cfg
	nice -n10\
		python -m luigi\
		--module detect Detect\
		--FileID "20210208_smGCN4_3h_B_12"\
		--ChannelIndex 0
	ls -lh /tungstenfs/scratch/gchao/eichbast/tests/fish_2d_analysis/*

three:
	cp /tungstenfs/scratch/gchao/eichbast/cuddly-octo-enigma/src/fish_3d.cfg /tungstenfs/scratch/gchao/eichbast/cuddly-octo-enigma/src/luigi.cfg
	nice -n10\
		python -m luigi\
		--module track Track\
		--FileID "20211117_GPAA1-565_SC-235-647_1h_19"\
		--ChannelIndex 0
	ls -lh /tungstenfs/scratch/gchao/eichbast/tests/fish_3d_analysis/*
