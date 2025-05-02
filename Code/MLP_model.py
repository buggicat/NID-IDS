import importlib
import os

def get_user_choice(prompt, choices):
    while True:
        print(prompt)
        for key, value in choices.items():
            print(f"{key}. {value}")

        try:
            choice = input("Enter your choice: ")
        except OSError:
            print("Input error. Please try again.")

        if choice in choices:
            return choices.get(choice, None)
        else:
            print("Invalid choice. Please try again.")

def main():
    while True:    
        mode_choices = {"1":"Feature Extraction", "2": "Feature Selection", "3": "Train the Model", "4": "Test the Model", "5": "Exit"}
        mode = get_user_choice("Select Mode:", mode_choices)

        if mode == "Exit":
            print("Exiting...")
            break
        
        if mode == "Feature Extraction":
            print(f"Running feature extraction for all features...")
            try:
                module = importlib.import_module("feature_extraction")
                # Store the original sys.argv
                import sys
                original_argv = sys.argv.copy()
                
                # Always extract all features
                os.environ['ATTACK_TYPE'] = "any"
                
                # Call the main function directly
                module.main()
                
                # Restore original sys.argv
                sys.argv = original_argv
            except Exception as e:
                print(f"Error during feature extraction: {e}")
            continue
        
        # For other modes, use the appropriate module
        if mode == "Feature Selection":
            module_name = "feature_selection"  
        elif mode == "Train the Model": 
            module_name = "model_train"
        elif mode == "Test the Model":
            module_name = "model_test"
            
        print(f"Running {module_name}...")
        try:
            module = importlib.import_module(module_name)
            module.main()
        except Exception as e:
            print(f"Error loading module: {e}")
    
if __name__ == "__main__":
    main()