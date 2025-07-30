########## Loss term experiments ###########
for rwm_factor in 0.1 1 10; do
    for energy_factor in 0 0.1 1 10; do
        for length_factor in 0 0.1 1 10; do
            if [ "$rwm_factor" = "$energy_factor" ] && [ "$rwm_factor" = "$length_factor" ] && [ "$rwm_factor" != 1 ]; then
                continue
            fi

            # if [ "$rwm_factor" = -1 ] ; then
            #     forecast=1
            #     context=1
            #     rwm_factor=1
            # else
                forecast=32
                context=32
            # fi

            echo "rwm_factor=$rwm_factor, energy_factor=$energy_factor, length_factor=$length_factor"
            python3 train_rwm.py --model unstructured --rwm_factor $rwm_factor --energy_factor $energy_factor --length_factor $length_factor --context $context --forecast $forecast --hidden_dim 64 --epochs 300 --batch_size 32
        done
    done
done

