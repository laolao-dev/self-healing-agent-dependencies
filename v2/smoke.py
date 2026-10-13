import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from scale_validation import make_corpus,run_suite,MUTATIONS,build_manifest,manifest_changes,dependency_index,affected_cases,failed_ids
cases=make_corpus()
base=run_suite(cases)
assert base['failed']==0,base
index=dependency_index(base)
results=[]
for mutation in MUTATIONS:
    changes=manifest_changes(build_manifest(),build_manifest(mutation))
    selected=affected_cases(cases,changes,index)
    full=run_suite(cases,mutation)
    target=run_suite(selected,mutation)
    assert failed_ids(full)==failed_ids(target) and failed_ids(full)
    results.append({'mutation':mutation.mutation_id,'full_count':len(cases),'targeted_count':len(selected),'caught':len(failed_ids(target))})
print(json.dumps({'baseline_passed':base['passed'],'mutations':results,'mode':'deterministic replay; no paid inference','abl_verification':'NOT_RUN'},indent=2))
