#!/bin/bash

under="_"
case=3
narrow=0


for stimDur in 50 100 200 400 800
	do
	for NneurStim in 100 150 200 250 300
	do
		for HAGA in 1 0
		do
			for relprob_dist in 1 0
			do
				nExperimentsNow=$(ps -ef | grep -v grep | grep run_experiment.py | wc -l)
				echo "nExperimentsNow = "$nExperimentsNow
				
				# make sure you don't run more than 10 experiments at the same time
				while [ $nExperimentsNow -ge 10 ]
				do
					nExperimentsNow=$(ps -ef | grep -v grep | grep run_experiment.py | wc -l)
					echo "nExperimentsNow = "$nExperimentsNow" WAITING for 10 s..."
					sleep 10
				done

				echo $relprob_dist $HAGA
				sleep 2
				if [ $HAGA -eq 0 ]; then
		 			suffix0="woFD"
				else
					suffix0="withFD"
				fi

				if [ $relprob_dist -eq 0 ]; then
		 			suffix1="fixedU"
				else
					suffix1="wideU"
				fi

				target_folder=experimentG$under$suffix0$under$suffix1$under$stimDur$under$NneurStim
				mkdir $target_folder
				cp run_experiment.py $target_folder
				cp bmm.dylib $target_folder
				cp cClasses_10b.py $target_folder
				cp df.pkl $target_folder
				cd $target_folder
				python run_experiment.py $relprob_dist $case $narrow $HAGA $NneurStim $stimDur &
				cd ..
			done
		done
	done
done