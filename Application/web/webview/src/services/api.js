import axios from 'axios';

// Create an axios instance
const service = axios.create({
  baseURL: 'http://localhost:5173', // Your API base path
  timeout: 5000, // Request timeout duration
});

// Request interceptor
service.interceptors.request.use(
  config => {
    // Do something before sending the request, e.g., add a token
    // config.headers['Authorization'] = `Bearer ${store.state.token}`;
    return config;
  },
  error => {
    // Handle request error
    return Promise.reject(error);
  }
);

// Response interceptor
service.interceptors.response.use(
  response => {
    // Do something with the response data
    const res = response.data;
    // console.log(response)
    if (response.status !== 200) {
      console.log("response.status !== 200")
      // Assuming there's a code field in the response data indicating error or success
      // You can handle it based on res.code
      return Promise.reject(new Error(res.message || 'Error'));
    } else {
      return res;
    }
  },
  error => {
    console.log("error process")
    // Handle response error
    return Promise.reject(error);
  }
);

// Encapsulate GET request
export const get = (url, params = {}) => {
  return service.get(url, { params });
};

// Encapsulate POST request
export const post = (url, data = {}) => {
  return service.post(url, data);
};

// Encapsulate PUT request
export const put = (url, data = {}) => {
  return service.put(url, data);
};

// Encapsulate DELETE request
export const del = (url, params = {}) => {
  return service.delete(url, { params });
};
