#!/bin/csh

foreach train_itr (100)
foreach train_update (10) 

@ ens = 1
while ( $ens <= 60)

echo "________________________________________________________________"
echo ""
echo "* train-iterations : "$train_itr
echo "* train-updates : "$train_update
echo "* # of ensembles : "$ens""
echo "________________________________________________________________"

sed "s/ENSEMBLE/$ens/g" setup_MAML.py > tmp1
sed "s/NUM_TRAIN_ITER/$train_itr/g" tmp1 > tmp2
sed "s/NUM_TRAIN_UPDATES/$train_update/g" tmp2 > MAML_run.py

python MAML_run.py

rm -f tmp1 tmp2
rm -f MAML_run.py

@ ens = $ens + 1; end

end #train_update
end #train_itr
