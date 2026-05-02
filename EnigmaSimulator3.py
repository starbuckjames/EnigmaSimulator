
#Engima Machine Simulator

rotor_1_string = "EKMFLGDQVZNTOWYHXUSPAIBRCJ"
rotor_2_string = "AJDKSIRUXBLHWTMCQGZNPYFVOE"
rotor_3_string = "BDFHJLCPRTXVZNYEIWGAKMUSQO"
rotor_4_string = "ESOVPZJAYQUIRHXLNFTGKDCMWB"
rotor_5_string = "VZBRGITYUPSDNHLXAWMJQOFECK"

etw_string = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
reflector_b_string = "YRUHQSLDPXNGOKMIEBFZCWVJAT"
reflector_c_string = "FVPJIAOYEDRZXWGCTKUQSBNMHL"

rotor_1_array = list(rotor_1_string)
rotor_2_array = list(rotor_2_string)
rotor_3_array = list(rotor_3_string)
rotor_4_array = list(rotor_4_string)
rotor_5_array = list(rotor_5_string)

etw_array = list(etw_string)
reflector_b_array = list(reflector_b_string)
reflector_c_array = list(reflector_c_string)

rotor_1_notch = 16
rotor_2_notch = 4
rotor_3_notch = 21
rotor_4_notch = 9
rotor_5_notch = 25

default_plugboard = {
    "A": "R",
    "B": "S",
    "C": "T",
    "D": "U",
    "E": "Q",
    "F": "P",
    "G": "J",
    "H": "Y",
    "I": "L",
    "K": "N",
    "R": "A",
    "S": "B",
    "T": "C",
    "U": "D",
    "Q": "E",
    "P": "F",
    "J": "G",
    "Y": "H",
    "L": "I",
    "N": "K"
}

plugboard = default_plugboard.copy()

def encrypt_message(message, rotor_right_start = 0, rotor_middle_start = 0, rotor_left_start = 0, rotor_right = rotor_1_array, rotor_middle = rotor_2_array, rotor_left = rotor_3_array, plugboard=plugboard):
    message = message.upper()
    message_list = list(message)
    message_encrypted = ""
    rotor_right_position = rotor_right_start
    rotor_middle_position = rotor_middle_start
    rotor_left_position = rotor_left_start

    if rotor_right == rotor_1_array:
        rotor_right_notch = rotor_1_notch
    elif rotor_right == rotor_2_array:
        rotor_right_notch = rotor_2_notch
    elif rotor_right == rotor_3_array:
        rotor_right_notch = rotor_3_notch
    elif rotor_right == rotor_4_array:
        rotor_right_notch = rotor_4_notch
    elif rotor_right == rotor_5_array:
        rotor_right_notch = rotor_5_notch

    if rotor_middle == rotor_1_array:
        rotor_middle_notch = rotor_1_notch
    elif rotor_middle == rotor_2_array:
        rotor_middle_notch = rotor_2_notch
    elif rotor_middle == rotor_3_array:
        rotor_middle_notch = rotor_3_notch
    elif rotor_middle == rotor_4_array:
        rotor_middle_notch = rotor_4_notch
    elif rotor_middle == rotor_5_array:
        rotor_middle_notch = rotor_5_notch



    for letter in message_list:
        if letter in etw_array:
            # Rotate rotors
        
            # 1. We must remember if the Middle Rotor was ALREADY at its notch 
            # BEFORE we move the Right Rotor. This is the secret to the Double Step.
            right_at_notch = (rotor_right_position == rotor_right_notch)
            middle_is_at_notch = (rotor_middle_position == rotor_middle_notch)

            # 2. Right rotor always moves every keypress
            rotor_right_position = (rotor_right_position + 1) % 26
            
            if right_at_notch or middle_is_at_notch:
                rotor_middle_position = (rotor_middle_position + 1) % 26

                if middle_is_at_notch:
                    rotor_left_position = (rotor_left_position + 1) % 26
                
            #Step 1: Plugboard swap
            letter = plugboard_swap(letter, plugboard)

            # Step 2: Pass through rotors (right to left)
            letter_index = etw_array.index(letter)

            # Right Rotor
            letter_index = (letter_index + rotor_right_position) % 26
            letter = rotor_right[letter_index]
            letter_index = (etw_array.index(letter) - rotor_right_position) % 26

            # Middle Rotor
            letter_index = (letter_index + rotor_middle_position) % 26
            letter = rotor_middle[letter_index]
            letter_index = (etw_array.index(letter) - rotor_middle_position) % 26

            # Left Rotor
            letter_index = (letter_index + rotor_left_position) % 26
            letter = rotor_left[letter_index]
            letter_index = (etw_array.index(letter) - rotor_left_position) % 26

            # Step 3: Reflector
            letter = etw_array[letter_index] # Signal enters reflector from current index
            letter_index = etw_array.index(letter)
            letter = reflector_b_array[letter_index]

            # Step 4: Pass back through rotors (left to right)
            letter_index = etw_array.index(letter)

            # Left Rotor (In reverse)
            letter_index = (letter_index + rotor_left_position) % 26
            letter = etw_array[letter_index]
            letter_index = (rotor_left.index(letter) - rotor_left_position) % 26

            # Middle Rotor (In reverse)
            letter_index = (letter_index + rotor_middle_position) % 26
            letter = etw_array[letter_index]
            letter_index = (rotor_middle.index(letter) - rotor_middle_position) % 26

            # Right Rotor (In reverse)
            letter_index = (letter_index + rotor_right_position) % 26
            letter = etw_array[letter_index]
            letter_index = (rotor_right.index(letter) - rotor_right_position) % 26
            
            letter = etw_array[letter_index]

            #Step 5: Plugboard swap again
            letter = plugboard_swap(letter, plugboard)

        message_encrypted += letter

       
        
    return message_encrypted

def decrypt_message(message, rotor_right_start = 0, rotor_middle_start = 0, rotor_left_start = 0, rotor_right = rotor_1_array, rotor_middle = rotor_2_array, rotor_left = rotor_3_array, plugboard=plugboard):
    #Decryption is the same as encryption, just with the rotors in reverse order
    return encrypt_message(message, rotor_right_start, rotor_middle_start, rotor_left_start, rotor_right, rotor_middle, rotor_left, plugboard)

def plugboard_swap(letter, plugboard):
    if letter in plugboard:
        return plugboard[letter]
    else:
        return letter
    
def change_plugboard_settings(plugboard):
    print("Current plugboard settings:")
    for key, value in plugboard.items():
        print(f"{key} <-> {value}")
    print("Enter new plugboard settings.  You must enter 10 pairs of letters (20 letters total). Example: A R B S C T D U E Q F P G J H Y I L K N")
    new_settings = input("New plugboard settings: ")
    new_settings_list = new_settings.split()
    plugboard.clear() # Clear the existing plugboard settings

    # Update the plugboard with the new settings
    for i in range(0, len(new_settings_list), 2):
        if i + 1 < len(new_settings_list):
            key = new_settings_list[i]
            value = new_settings_list[i + 1]
            plugboard[key] = value
        
    # Adds the reverse mappings to the plugboard
    reverse_mappings = {}
    for key, value in plugboard.items():
        reverse_mappings[value] = key
    plugboard.update(reverse_mappings)
    #prints the new plugboard settings
    print("New plugboard settings:")
    for key, value in plugboard.items():
        print(f"{key} <-> {value}")

def show_current_settings(rotor_right, rotor_middle, rotor_left, plugboard):
    print("Current rotor settings:")
    print(f"Right Rotor: {rotor_right}")
    print(f"Middle Rotor: {rotor_middle}")
    print(f"Left Rotor: {rotor_left}")
    print("Current plugboard settings:")
    for key, value in plugboard.items():
        print(f"{key} <-> {value}")

def main_menu():
    print("Welcome to the Enigma Machine Simulator!")
    print("1. Encrypt a message")
    print("2. Decrypt a message")
    print("3. Change plugboard settings")
    print("4. Change rotor settings")
    print("5. Show Current Settings")
    print("6. Reset to Default Settings")
    print("7. Exit")

if __name__ == "__main__":
    active = True
    rotor_right = rotor_1_array
    rotor_middle = rotor_2_array
    rotor_left = rotor_3_array

    while active == True:
        main_menu()
        choice = input("Enter your choice: ")
        if choice == "1":
            message = input("Enter the message to encrypt: ")
            rotor_right_start = int(input("Enter the starting position for the right rotor (0-25): "))
            rotor_middle_start = int(input("Enter the starting position for the middle rotor (0-25): "))
            rotor_left_start = int(input("Enter the starting position for the left rotor (0-25): "))
            encrypted_message = encrypt_message(message, rotor_right_start, rotor_middle_start, rotor_left_start, rotor_right, rotor_middle, rotor_left, plugboard)
            print("Encrypted message: " + encrypted_message)
        elif choice == "2":
            message = input("Enter the message to decrypt: ")
            rotor_right_start = int(input("Enter the starting position for the right rotor (0-25): "))
            rotor_middle_start = int(input("Enter the starting position for the middle rotor (0-25): "))
            rotor_left_start = int(input("Enter the starting position for the left rotor (0-25): "))
            decrypted_message = decrypt_message(message, rotor_right_start, rotor_middle_start, rotor_left_start, rotor_right, rotor_middle, rotor_left, plugboard)
            print("Decrypted message: " + decrypted_message)
        elif choice == "3":
            change_plugboard_settings(plugboard)
        elif choice == "4":
            print("Current rotor settings:")
            print(f"Right Rotor: {rotor_right}")
            print(f"Middle Rotor: {rotor_middle}")
            print(f"Left Rotor: {rotor_left}")
            print("Available rotors:")
            print("1. Rotor 1")
            print("2. Rotor 2")
            print("3. Rotor 3")
            print("4. Rotor 4")
            print("5. Rotor 5")
            right_choice = input("Choose the right rotor (1-5): ")
            middle_choice = input("Choose the middle rotor (1-5): ")
            left_choice = input("Choose the left rotor (1-5): ")

            rotor_choices = {
                "1": rotor_1_array,
                "2": rotor_2_array,
                "3": rotor_3_array,
                "4": rotor_4_array,
                "5": rotor_5_array
            }

            if right_choice in rotor_choices:
                rotor_right = rotor_choices[right_choice]
            if middle_choice in rotor_choices:
                rotor_middle = rotor_choices[middle_choice]
            if left_choice in rotor_choices:
                rotor_left = rotor_choices[left_choice]

            print("New rotor settings:")
            print(f"Right Rotor: {rotor_right}")
            print(f"Middle Rotor: {rotor_middle}")
            print(f"Left Rotor: {rotor_left}")
        elif choice == "5":
            show_current_settings(rotor_right, rotor_middle, rotor_left, plugboard)
        elif choice == "6":
            rotor_right = rotor_1_array
            rotor_middle = rotor_2_array
            rotor_left = rotor_3_array
            plugboard = default_plugboard.copy()
            print("Settings reset to default.")
        elif choice == "7":
            active = False
        else:
            print("Invalid choice. Please try again.")