# Teeth authentification
Using the position and the shape of teeth as an authentication token

The program selects a contour that contains information about the location and shape of teeth, and then compares this contour with the contours stored in the users folder

All user information is encrypted
<!--The principle of operation-->
## Manual
### Settings
| Name          | Description                                                     |
|---------------|-----------------------------------------------------------------|
| key  	        | The encryption key (must be 16 bytes)                           |
| webcum_input  | Webcam index (use "None" to work with images)                                      |
| image_path       | The path to the image                              |
| users_dir	   | The path to the user directory                                         |
| image_error	   | The path to the error message image                                   |
| find_mouth_flag	   | The flag for using a neural network to find the used area of the image (mouth)                                              |
| bilateral_filter_flag	   | Flag for using a Bilateral filter instead of a Gaussian filter (depending on shooting conditions)          |
| debug_save_flag | Flag to enable saving images for debugging |
| debug_save_path   | The path to save the debug files                                    |

### Hotkeys
'q' - to exit the program

'w' - to write the new user

In case of successful authentication, the program will exit with a message about the user name

### Using a previusly made image
The image should contain an image with the mouth slightly open in the direction of the front

<p align="center">
  <img src="https://github.com/porodzinskiy/teeth-biometrics/blob/master/images/user.jpg?raw=true" width="350" title="An example">
</p>

#### Recommended settings: 
``` webcum_input = None ```
``` find_mouth_flag = False ```
``` bilateral_filter_flag = True ```

### Using webcam capture
The user should be at a distance of about 30-50 cm from the webcam and slightly open his mouth so that his teeth are visible, as in the example above

The lighting should be adequate

It is recommended to use automatic allocation of the working area to avoid scanning errors

#### Recommended settings:
``` webcum_input = 0 (or your webcam index) ```
``` find_mouth_flag = True ```
``` bilateral_filter_flag = True ```

### Libraries used
OpenCV
``` pip install opencv-python ```

NumPy
``` pip install numpy ```

PyCryptodome
``` pip install pycryptodome ```


## Tests
The use of neural networks to automatically search for a workspace slows down the system by 128 percent

FRR (False Rejection Rate) = 1.6 %

FAR (False Acceptance Rate) = 0.8 %
