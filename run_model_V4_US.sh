git config --global user.email "ali@dynideas.com"; git config --global user.name "Server" ;
cd ..;cd ..;cd /repos/DELPHI
python DELPHI_model_V3_predict_additional_state.py -d "2020-10-01" -t 'US'

python DELPHI_model_V4_additional_states.py -rc run_configs/annealing-run-config.yml -t 'US'

git stash -- data_sandbox/predicted/raw_predictions/*
git stash -- data_sandbox/predicted/policy_scenario_predictions/*
git stash -- data_sandbox/processed/*
git stash -- data_sandbox/raw_data_additional_states/*
git stash -- danger_map/*
git stash -- logs/*

git pull

git add data_sandbox/predicted/parameters/*
#git add data_sandbox/predicted/policy_scenario_predictions/*
git -c user.name='Server' -c user.email=ali@dynideas.com commit -m "US results"
git push https://"$GIT_USER":"$GIT_PASS"@github.com/COVIDAnalytics/DELPHI.git --all

