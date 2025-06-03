مشروع تصنيف الأرقام باستخدام شبكة عصبية
هذا المشروع يستخدم مكتبة TensorFlow لتصميم نموذج شبكة عصبية لتصنيف صور الأرقام المكتوبة يدويًا (0 إلى 9) من قاعدة بيانات MNIST.

المتطلبات
Python 3.7 أو أحدث

المكتبات المطلوبة:

numpy

matplotlib

tensorflow

لتثبيت المكتبات:


pip install numpy matplotlib tensorflow
وصف البيانات
يتم تحميل بيانات MNIST مباشرة من مكتبة Keras. تحتوي على:

60000 صورة للتدريب

10000 صورة للاختبار

حجم الصورة: 28 × 28 بكسل، تدرج رمادي

التصنيفات من 0 إلى 9

خطوات التنفيذ
تحميل البيانات من مكتبة Keras

عرض توزيع البيانات باستخدام مخطط دائري لكل رقم

عرض صور مختارة من الرقم 4 (25 صورة)

تطبيع الصور بقسمة القيم على 255

تصميم نموذج الشبكة العصبية يتكون من:

طبقة Flatten لتحويل الصورة إلى متجه

3 طبقات Dense (512، 256، 128 وحدة) مع تفعيل ReLU

طبقة إخراج بـ 10 وحدات مع تفعيل softmax

تجميع النموذج باستخدام خوارزمية Adam وخسارة تصنيف متعددة الفئات

تدريب النموذج على بيانات التدريب لمدة 10 تكرارات

تقييم النموذج على بيانات الاختبار

رسم صور من مجموعة الاختبار مع التنبؤات والتلوين حسب صحة التنبؤ

النتائج
دقة النموذج على بيانات الاختبار يتم طباعتها

عرض صور من بيانات الاختبار مع التنبؤ الحقيقي والمتوقع وتلوين الاسم بلون أخضر (صحيح) أو أحمر (خاطئ)


Handwritten Digit Classification using Neural Network
This project uses TensorFlow to build a neural network model for classifying handwritten digits (0 to 9) from the MNIST dataset.

Requirements
Python 3.7 or later

Required libraries:

numpy

matplotlib

tensorflow

Install using:

pip install numpy matplotlib tensorflow
Dataset
MNIST dataset loaded from Keras

60000 training images

10000 test images

Image size: 28 × 28 pixels, grayscale

Labels: digits from 0 to 9

Workflow
Load the dataset from Keras

Visualize class distribution using a pie chart

Display 25 samples of digit "4"

Normalize the images by dividing pixel values by 255

Build the neural network:

Flatten layer to convert image to vector

Dense layers: 512, 256, 128 neurons with ReLU

Output layer: 10 neurons with softmax

Compile the model with Adam optimizer and sparse categorical crossentropy loss

Train the model for 10 epochs

Evaluate the model on test data

Plot test images with predictions and highlight correct (green) and incorrect (red)

Output
Test accuracy is printed

25 sample test images are plotted with predicted and actual labels

