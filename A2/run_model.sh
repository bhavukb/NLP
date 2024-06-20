if [ $1="test" ]
then
    python testing.py $2 $3
else
    python initial.py $2 $3
fi
