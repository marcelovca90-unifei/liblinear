#!/bin/bash

JVM_OPTS="-Xmx8G -Xms80m"
METADATA="/Users/marcelocysneiros/git/marcelovca90-unifei/anti-spam-weka-data/metadata/_RELEVANT.txt"
PRIMES=(2 3 5 7 11 13 17 19 23 29)
KERNELS=("poly1" "poly2")
SOLVERS=("solver1")

for KERNEL in "${KERNELS[@]}"
do
    for SOLVER in "${SOLVERS[@]}"
    do
        cat "header.log" > "$KERNEL$SOLVER.log"
        while read p;
        do
            FOLDER=$(echo $p | cut -d',' -f1 | sed -e "s/~/\/Users\/marcelocysneiros/g")
            EMPTY_HAM_COUNT=$(echo $p | cut -d',' -f2)
            EMPTY_SPAM_COUNT=$(echo $p | cut -d',' -f3)
            echo "$FOLDER | $KERNEL | $SOLVER"

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
            java $JVM_OPTS -jar ./arff2liblinear.jar aggregate $FOLDER/data.partial_results $FOLDER/data.train_times $FOLDER/data.test_times >> "$KERNEL$SOLVER.log"

            # tear down
            cd $FOLDER && ls $FOLDER | grep -v arff | grep -v best_c | xargs rm && cd - > /dev/null
              
        done <$METADATA
    done
done
