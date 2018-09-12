#!/bin/bash
TAG=$1
echo $TAG
read -p "Do you want to continue? Y / n " input
if [ $input == "Y" ]
then
    rm -r log/$TAG
    rm -r save_model/$TAG
    rm -r predicts/$TAG
    rm -r predicts-all/$TAG
else
    exit
fi