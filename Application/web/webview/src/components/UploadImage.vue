<template>
  <div class="body">
    <div class="imageForm">
      <div v-if="imagePreview">
        <h3>Preview of the selected image:</h3>
        <img :src="imagePreview" alt="Selected Image" style="max-width: 400px; height: auto;" />
      </div>
      <div class="response" v-if="responseData">
        <h3>Class: {{ responseData.class }}</h3>
        <h3 v-if="parseFloat(responseData.confidence) >= 80">Confidence: {{ responseData.confidence }}</h3>
      </div>
    </div>
    <div class="footer">
      <el-upload class="upload-demo" action="" :show-file-list="false" :before-upload="handleBeforeUpload">
        <el-button size="default" type="primary">Upload image</el-button>
      </el-upload>
      <el-button  :disabled="!imagePreview" type="success" @click="uploadImage">Classification</el-button>
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
      responseData: {
        class: "NULL",
        confidence: "NULL"

      }
    };
  },
  methods: {
    handleBeforeUpload(file) {
      const isImage  = file.type === 'image/jpeg' || file.type === 'image/png';
      if (!isImage) {
        this.$message.error('Can upload JPG or PNG only!');
        return false; 
      }

      this.selectedFile = file;

      const reader = new FileReader();
      reader.readAsDataURL(file);

      reader.onload = () => {
        this.imagePreview = reader.result;
      };

      return false; 
    },
    async uploadImage() {
      if (!this.selectedFile) return;

      const reader = new FileReader();
      reader.readAsDataURL(this.selectedFile);

      reader.onload = async () => {
        console.log(reader)
        const base64Image = reader.result;
        const cleanedBase64String = base64Image.replace(/^data:image\/\w+;base64,/, '');

        try {
          // Using the encapsulated POST method
          const response = await post('/api/classify', { image: cleanedBase64String });
          response.confidence = (response.confidence * 100).toFixed(2) + '%';
          this.responseData = response
          console.log(response)
        } catch (error) {
          console.error('Failed to upload image:', error);
        }
      };
    },
  },
};
</script>

<style>
.imageForm {
  display: flex;
  flex-direction: row;
  gap: 10px;
}

.response {
  width: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin-left: 20px;
}
.footer{
  margin-top: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;

}
.upload-demo {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 0 !important;
}


</style>