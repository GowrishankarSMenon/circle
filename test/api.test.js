// Use CommonJS requires instead of ES modules
const chai = require('chai');
const chaiHttp = require('chai-http');
const fs = require('fs');
const path = require('path');

// Configure Chai
chai.use(chaiHttp);
const expect = chai.expect;
const should = chai.should();

// Base URL of your Flask API
const BASE_URL = 'http://localhost:5000';

describe('Waste Classification API', function() {
  // Increase timeout for tests that might take longer
  this.timeout(5000);

  // Test image path
  const testImagePath = path.join(__dirname, 'test.jpg');
  
  before(function() {
    // Create a dummy test image if it doesn't exist
    if (!fs.existsSync(testImagePath)) {
      fs.writeFileSync(testImagePath, Buffer.from([0xFF, 0xD8, 0xFF, 0xE0])); // Minimal JPEG header
    }
  });

  describe('GET /', function() {
    it('should return API documentation', function(done) {
      chai.request(BASE_URL)
        .get('/')
        .end(function(err, res) {
          if (err) return done(err);
          res.should.have.status(200);
          res.text.should.include('Recyclable Waste Classification API');
          done();
        });
    });
  });

  describe('POST /classify', function() {
    it('should classify a single image', function(done) {
      chai.request(BASE_URL)
        .post('/classify')
        .attach('image', fs.readFileSync(testImagePath), 'test.jpg')
        .end(function(err, res) {
          if (err) return done(err);
          res.should.have.status(200);
          res.body.should.be.an('object');
          res.body.should.have.property('status');
          if (res.body.status === 'success') {
            res.body.should.have.property('detected_objects').that.is.an('array');
            res.body.should.have.property('summary');
          }
          done();
        });
    });

    it('should reject invalid file types', function(done) {
      chai.request(BASE_URL)
        .post('/classify')
        .attach('image', fs.readFileSync(__filename), 'invalid.txt')
        .end(function(err, res) {
          res.should.have.status(400);
          res.body.should.have.property('error');
          res.body.error.should.include('File type not allowed');
          done();
        });
    });
  });

  describe('GET /categories', function() {
    it('should return list of supported categories', function(done) {
      chai.request(BASE_URL)
        .get('/categories')
        .end(function(err, res) {
          if (err) return done(err);
          res.should.have.status(200);
          res.body.should.have.property('categories').that.is.an('array');
          res.body.categories.should.include('bottle');
          done();
        });
    });
  });

  describe('POST /batch_classify', function() {
    it('should classify multiple images', function(done) {
      chai.request(BASE_URL)
        .post('/batch_classify')
        .attach('images', fs.readFileSync(testImagePath), 'test1.jpg')
        .attach('images', fs.readFileSync(testImagePath), 'test2.jpg')
        .end(function(err, res) {
          if (err) return done(err);
          res.should.have.status(200);
          res.body.should.have.property('results').that.is.an('array');
          res.body.results.should.have.lengthOf(2);
          done();
        });
    });
  });

  describe('POST /add_mapping', function() {
    it('should add custom object mapping', function(done) {
      chai.request(BASE_URL)
        .post('/add_mapping')
        .send({
          object_class: 'test_object',
          recyclability: 'Recyclable'
        })
        .end(function(err, res) {
          if (err) return done(err);
          res.should.have.status(200);
          res.body.should.have.property('message');
          res.body.message.should.include('Added mapping');
          done();
        });
    });

    it('should reject invalid recyclability', function(done) {
      chai.request(BASE_URL)
        .post('/add_mapping')
        .send({
          object_class: 'test_object',
          recyclability: 'InvalidCategory'
        })
        .end(function(err, res) {
          res.should.have.status(400);
          done();
        });
    });
  });

  // Clean up test file
  after(function() {
    if (fs.existsSync(testImagePath)) {
      fs.unlinkSync(testImagePath);
    }
  });
});