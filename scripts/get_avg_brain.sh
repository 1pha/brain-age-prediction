for mode in train valid test
do
    python -m fire sage.xai.average get_ukb_average --mode=$mode
done
