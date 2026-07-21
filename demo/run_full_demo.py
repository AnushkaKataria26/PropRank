import sys
from demo.setup_demo_data import setup_demo_data
from demo.scenario_1_cold_start import run_scenario_1
from demo.scenario_2_warm_context import run_scenario_2
from demo.scenario_3_retraining import run_scenario_3

def main():
    print("=======================================")
    print("      PropRank End-to-End Demo         ")
    print("=======================================\n")
    
    print("--- Setup Demo Data ---")
    try:
        setup_demo_data()
    except Exception as e:
        print(f"Demo setup failed: {e}")
        sys.exit(1)
        
    results = []
    
    s1 = run_scenario_1()
    results.append(("Scenario 1 (Cold Start)", s1))
    
    s2 = run_scenario_2()
    results.append(("Scenario 2 (Warm Context)", s2))
    
    # Scenario 3 depends on Scenario 2 for an active model
    if s2:
        s3 = run_scenario_3()
        results.append(("Scenario 3 (Retraining)", s3))
    else:
        results.append(("Scenario 3 (Retraining)", False))
        print("Skipping Scenario 3 because Scenario 2 failed to produce an active model.")
        
    print("=======================================")
    print("            Demo Summary               ")
    print("=======================================")
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<30} | {status}")
        if not passed:
            all_passed = False
            
    print("=======================================")
    
    if not all_passed:
        print("Demo completed with failures.")
        sys.exit(1)
    else:
        print("Demo completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
