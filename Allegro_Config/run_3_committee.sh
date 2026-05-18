#!/bin/bash
# Copyright (c) 2025 Christopher T. Davies
# Licensed under the Academic Software License v1.0 (ASL)

CONFIG_PATH="{HPC_PATH}/training/active_learning_train"
CONFIG_NAME="config.yaml"
GENERATION="17"
FOLDER_PREFIX="gen"
COMPILE=true

export NEQUIP_FLOAT32_MODEL_TOL=0.001
export PYTHONPATH="$PYTHONPATH:$CONFIG_PATH"

# each seeed is a different random seed for training 
SEEDS=(12 13 14)

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    NAME="ft_isom_g${GENERATION}_$i"
    FOLDER_NAME="${FOLDER_PREFIX}_${GENERATION}_$i"

    echo "Launching: Seed $SEED, Name $NAME, Folder $FOLDER_NAME"
    mkdir -p "$FOLDER_NAME"

    nequip-train -cp "$CONFIG_PATH" \
        -cn "$CONFIG_NAME" \
        hydra.run.dir="$FOLDER_NAME" \
        data.seed=$SEED \
        trainer.logger.name="$NAME" > "${FOLDER_NAME}/${NAME}.log" 2>&1 &

    sleep 5
done

wait
echo "All training runs completed."

if [ "$COMPILE" = true ]; then
    echo "Starting compilation..."

    for i in "${!SEEDS[@]}"; do
        NAME="ft_isom_g${GENERATION}_$i"
        FOLDER_NAME="${FOLDER_PREFIX}_${GENERATION}_$i"

        if [ -d "$FOLDER_NAME" ]; then
            CKPT_PATH=$(find "$FOLDER_NAME" -name "epoch*.ckpt" -print -quit)

            if [ -n "$CKPT_PATH" ]; then
                echo "Compiling $NAME from $CKPT_PATH"
                OUTPUT_PATH="${CKPT_PATH}.ase.nequip.pt2"
                OUTPUT_PATH_LAMMPS="${CKPT_PATH}.lammps.nequip.pt2"    

                nequip-compile "$CKPT_PATH" "$OUTPUT_PATH" \
                    --device cuda --mode aotinductor --target ase \
                    --modifiers enable_TritonContracter >> "${FOLDER_NAME}/${NAME}_compile.log" 2>&1

                nequip-compile "$CKPT_PATH" "$OUTPUT_PATH_LAMMPS" \
                    --device cuda --mode aotinductor --target pair_allegro \
                    --modifiers enable_TritonContracter >> "${FOLDER_NAME}/${NAME}_compile.log" 2>&1
            else
                echo "WARNING: No checkpoint found for $NAME in $FOLDER_NAME"
            fi
        else
            echo "WARNING: Run directory $FOLDER_NAME does not exist."
        fi
    done
fi
