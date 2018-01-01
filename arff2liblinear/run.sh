#!/bin/bash

JVM_OPTS="-Xmx8G -Xms80m"
METADATA="/Users/marcelocysneiros/git/anti-spam-weka-data/2017_BASE2_ARFF/metadata.txt"
PRIMES=(2 3 5 7 11 13 17 19 23 29)
KERNEL="poly1"
SOLVER="solver1"

while read p; do
  FOLDER=$(echo $p | cut -d',' -f1 | sed -e "s/~/\/Users\/marcelocysneiros/g")
  EMPTY_HAM_COUNT=$(echo $p | cut -d',' -f2)
  EMPTY_SPAM_COUNT=$(echo $p | cut -d',' -f3)

  for SEED in "${PRIMES[@]}"
  do
      # prepare
      java $JVM_OPTS -jar ./arff2liblinear.jar prepare $FOLDER/data.arff $EMPTY_HAM_COUNT $EMPTY_SPAM_COUNT $SEED

      # train
      java $JVM_OPTS -jar ./arff2liblinear.jar train $KERNEL $SOLVER $FOLDER/data.train

      # test
      java $JVM_OPTS -jar ./arff2liblinear.jar test $KERNEL $FOLDER/data.test $FOLDER/data.model > /dev/null

      # evaluate
      java $JVM_OPTS -jar ./arff2liblinear.jar evaluate $FOLDER/data.test $FOLDER/data.prediction
  done
      # aggregate
      java $JVM_OPTS -jar ./arff2liblinear.jar aggregate $FOLDER/data.partial_results $FOLDER/data.train_times $FOLDER/data.test_times

      # tear down
      cd $FOLDER && ls $FOLDER | grep -v arff | xargs rm && cd - > /dev/null
done <$METADATA
