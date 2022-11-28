#!/bin/bash
cd .. && \
tar \
--exclude=*/BMM_SEQ_STIM/* \
--exclude=snapshots/* \
--exclude=data* \
--exclude=*.png \
--exclude=.git* \
-czvf snapshots/snap_$1.tar.gz .