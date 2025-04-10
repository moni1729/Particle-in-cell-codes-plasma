DIR=$(pwd)
mkdir -p $DIR/results/$1
cd simulations/$1
qpic-movie --beam 1 --xy-units um --z-units um --ignore-last --frames 100 --threads 4 --style magma --frames-folder "$DIR/results/$1/frames/drive" --out "$DIR/results/$1/drive.mp4" &
qpic-movie --beam 2 --xy-units um --z-units um --ignore-last --frames 100 --threads 4 --style magma --frames-folder "$DIR/results/$1/frames/witness" --out "$DIR/results/$1/witness.mp4" &
qpic-movie --species 1 --xy-units um --z-units um --ignore-last --frames 100 --threads 4 --style magma --frames-folder "$DIR/results/$1/frames/plasma" --out "$DIR/results/$1/plasma.mp4" &
!qpic-movie --field ez --xy-units um --z-units um --ignore-last --frames 100 --threads 4 --frames-folder "$DIR/results/$1/frames/ez" --out "$DIR/results/$1/ez.mp4" &
wait
