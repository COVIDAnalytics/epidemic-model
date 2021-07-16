git config --global user.email "ali@dynideas.com"; git config --global user.name "Server" ;
cd ..;cd ..;cd /repos/DELPHI
python DELPHI_model_V3_predict_additional_state.py -d "2020-10-01" -t 'ExUS' -g 'false'

python DELPHI_model_V4_additional_states.py -rc run_configs/annealing-run-config.yml -t 'ExUS' -g 'false' -weekago 'true'

#git stash
#git pull
##git add data_sandbox/predicted/policy_scenario_predictions/*
#git -c user.name='Server' -c user.email=ali@dynideas.com commit -m "Ex US a week ago results"
#git push https://"$GIT_USER":"$GIT_PASS"@github.com/COVIDAnalytics/DELPHI.git --all
#
