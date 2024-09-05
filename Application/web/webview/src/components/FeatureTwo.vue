<template>
  <div class="main">

    <div class="left">
      <div v-if="imagePreview">
        <h3>Preview of the selected image:</h3>
        <img :src="imagePreview" alt="Selected Image" style="max-width: 500px; height: auto;" />
      </div>
      <div class="leftBottom">
        <el-upload class="upload-demo" action="" :show-file-list="false" :before-upload="handleBeforeUpload">
          <el-button size="default" type="primary">Upload</el-button>
        </el-upload>
        <el-button v-if="imagePreview" type="success" @click="uploadImage">Segmentation</el-button>
      </div>

    </div>
    <div class="right">
      <div v-if="processedImage">
        <h3>The image after segmentation:</h3>
        <img :src="processedImage" alt="Processed Image" style="max-width: 500px; height: auto;" />
      </div>
      <div v-if="segmentationError">
        <h3>{{ segmentationError }}</h3>
      </div>
    </div>
  </div>

</template>

<script>
// Import the encapsulated request method
import { post } from '@/services/api';

export default {
  data() {
    return {
      selectedFile: null,
      imagePreview: null,
      processedImage: null,
      responseData: null,
      segmentationError: null // Add a new state to display error messages
    };
  },
  methods: {
    handleBeforeUpload(file) {
      const isImage  = file.type === 'image/jpeg' || file.type === 'image/png';
      if (!isImage ) {
        this.$message.error('Can upload JPG or PNG only!');
        return false; // Block upload
      }

      this.selectedFile = file;

      const reader = new FileReader();
      reader.readAsDataURL(file);

      reader.onload = () => {
        this.imagePreview = reader.result;
      };

      return false; // Block automatic uploads
    },
    async uploadImage() {
      if (!this.selectedFile) return;

      const reader = new FileReader();
      reader.readAsDataURL(this.selectedFile);

      reader.onload = async () => {
        const base64Image = reader.result;
        const cleanedBase64String = base64Image.replace(/^data:image\/\w+;base64,/, '');

        try {
          // Using the encapsulated POST method
          const response = await post('/api/segmentate', { image: cleanedBase64String });

          // Determine if the server returns "null"
          if (response.segmented_image === "null" ) {
            this.processedImage = null;
            this.segmentationError = "This image cannot be segmented.";
          } else {
            this.processedImage = "data:image/jpeg;base64," + response.segmented_image;
            this.segmentationError = null;
          }
        } catch (error) {
          console.error('Failed to upload image:', error);
          this.segmentationError = "Failed to process the image.";
        }
      };
    },
  },
};
</script>

<style>
* {
  padding: 0;
  margin: 0;
}

.main {
  height: 100%;
  display: flex;
  flex-direction: row;
  justify-content: center;
  gap: 10%;
}

.left {
  display: flex;
  flex-direction: column;
  gap: 10%;
  justify-content: center;
  align-items: center;
}

.leftBottom {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
}
.right {
  display: flex;
  justify-content: center;
  align-items: start;
}

.upload-demo {
  display: block;

}
</style>
