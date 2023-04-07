﻿using Microsoft.ML.OnnxRuntime.Tensors;
using MathNet.Numerics;
using NumSharp;

namespace StableDiffusion
{
    public class LMSDiscreteScheduler
    {
        private int _numTrainTimesteps;
        private string _predictionType;
        private List<float> _alphasCumulativeProducts;

        public Tensor<float> Sigmas;
        public List<int> Timesteps;
        public List<Tensor<float>> Derivatives;
        public float InitNoiseSigma;

        public LMSDiscreteScheduler(int num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, string beta_schedule = "scaled_linear", string prediction_type = "epsilon", List<float> trained_betas = null)
        {
            _numTrainTimesteps = num_train_timesteps;
            _predictionType = prediction_type;
            _alphasCumulativeProducts = new List<float>();
            Derivatives = new List<Tensor<float>>();
            Timesteps = new List<int>();

            var alphas = new List<float>();
            var betas = new List<float>();

            if (trained_betas != null)
            {
                betas = trained_betas;
            }
            else if (beta_schedule == "linear")
            {
                betas = Enumerable.Range(0, num_train_timesteps).Select(i => beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1)).ToList();
            }
            else if (beta_schedule == "scaled_linear")
            {
                var start = (float)Math.Sqrt(beta_start);
                var end = (float)Math.Sqrt(beta_end);
                betas = np.linspace(start, end, num_train_timesteps).ToArray<float>().Select(x => x * x).ToList();

            }
            else
            {
                throw new Exception("beta_schedule must be one of 'linear' or 'scaled_linear'");
            }

            alphas = betas.Select(beta => 1 - beta).ToList();
 
            this._alphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b)).ToList();
            // Create sigmas as a list and reverse it
            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();

            // standard deviation of the initial noise distrubution
            this.InitNoiseSigma = (float)sigmas.Max();

        }

        // Line 157 of scheduling_lms_discrete.py from HuggingFace diffusers
        public int[] SetTimesteps(int num_inference_steps)
        {
            double start = 0;
            double stop = _numTrainTimesteps - 1;
            double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

            this.Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
            sigmas = Interpolate(timesteps, range, sigmas).ToList();
            this.Sigmas = new DenseTensor<float>(sigmas.Count());
            for (int i = 0; i < sigmas.Count(); i++)
            {
                this.Sigmas[i] = (float)sigmas[i];
            }
            return this.Timesteps.ToArray();

        }

        public static double[] Interpolate(double[] timesteps, double[] range, List<double> sigmas)
        {

            // Create an output array with the same shape as timesteps
            var result = np.zeros(timesteps.Length + 1);

            // Loop over each element of timesteps
            for (int i = 0; i < timesteps.Length; i++)
            {
                // Find the index of the first element in range that is greater than or equal to timesteps[i]
                int index = Array.BinarySearch(range, timesteps[i]);

                // If timesteps[i] is exactly equal to an element in range, use the corresponding value in sigma
                if (index >= 0)
                {
                    result[i] = sigmas[index];
                }

                // If timesteps[i] is less than the first element in range, use the first value in sigmas
                else if (index == -1)
                {
                    result[i] = sigmas[0];
                }

                // If timesteps[i] is greater than the last element in range, use the last value in sigmas
                else if (index == -range.Length - 1)
                {
                    result[i] = sigmas[-1];
                }

                // Otherwise, interpolate linearly between two adjacent values in sigmas
                else
                {
                    index = ~index; // bitwise complement of j gives the insertion point of x[i]
                    double t = (timesteps[i] - range[index - 1]) / (range[index] - range[index - 1]); // fractional distance between two points
                    result[i] = sigmas[index - 1] + t * (sigmas[index] - sigmas[index - 1]); // linear interpolation formula
                }

            }
            //  add 0.000 to the end of the result
            result = np.add(result, 0.000f);

            return result.ToArray<double>();
        }

        public DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            // Get step index of timestep from TimeSteps
            int stepIndex = this.Timesteps.IndexOf(timestep);
            // Get sigma at stepIndex
            var sigma = this.Sigmas[stepIndex];
            sigma = (float)Math.Sqrt((Math.Pow(sigma, 2) + 1));

            // Divide sample tensor shape {2,4,64,64} by sigma
            sample = TensorHelper.DivideTensorByFloat(sample.ToArray(), sigma, sample.Dimensions.ToArray());

            return sample;
        }

        //python line 135 of scheduling_lms_discrete.py
        public double GetLmsCoefficient(int order, int stepIndex, int currOrder)
        {
            // Compute a linear multistep coefficient.

            double LmsDerivative(double tau)
            {
                double prod = 1.0;
                for (int k = 0; k < order; k++)
                {
                    if (currOrder == k)
                    {
                        continue;
                    }
                    prod *= (tau - this.Sigmas[stepIndex - k]) / (this.Sigmas[stepIndex - currOrder] - this.Sigmas[stepIndex - k]);
                }
                return prod;
            }

            double integratedCoeff = Integrate.OnClosedInterval(LmsDerivative, this.Sigmas[stepIndex], this.Sigmas[stepIndex + 1], 1e-4);

            return integratedCoeff;
        }

        public DenseTensor<float> Step(
               Tensor<float> noisePred,
               int timestep,
               Tensor<float> latents,
               int order = 4)
        {
            int stepIndex = this.Timesteps.IndexOf(timestep);
            var sigma = this.Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            Tensor<float> predOriginalSample;

            // Create array of type float length modelOutput.length
            float[] predOriginalSampleArray = new float[noisePred.Length];
            var noiseArray = noisePred.ToArray();
            var latentsArray = latents.ToArray();

            if (this._predictionType == "epsilon")
            {

                for (int i=0; i < noiseArray.Length; i++)
                {
                    predOriginalSampleArray[i] = latentsArray[i] - sigma * noiseArray[i];
                }
                predOriginalSample = TensorHelper.CreateTensor(predOriginalSampleArray, noisePred.Dimensions.ToArray());

            }
            else if (this._predictionType == "v_prediction")
            {
                //predOriginalSample = modelOutput * ((-sigma / Math.Sqrt((Math.Pow(sigma,2) + 1))) + (sample / (Math.Pow(sigma,2) + 1)));
                throw new Exception($"prediction_type given as {this._predictionType} not implemented yet.");
            }
            else
            {
                throw new Exception($"prediction_type given as {this._predictionType} must be one of `epsilon`, or `v_prediction`");
            }

            // 2. Convert to an ODE derivative
            var derivativeItems = new DenseTensor<float>(latents.Dimensions.ToArray());

            var derivativeItemsArray = new float[derivativeItems.Length];
            
            for (int i = 0; i < noiseArray.Length; i++)
            {
                //predOriginalSample = (sample - predOriginalSample) / sigma;
                derivativeItemsArray[i] = (latentsArray[i] - predOriginalSampleArray[i]) / sigma;
            }
            derivativeItems =  TensorHelper.CreateTensor(derivativeItemsArray, derivativeItems.Dimensions.ToArray());

            this.Derivatives?.Add(derivativeItems);

            if (this.Derivatives?.Count() > order)
            {
                // remove first element
                this.Derivatives?.RemoveAt(0);
            }

            // 3. compute linear multistep coefficients
            order = Math.Min(stepIndex + 1, order);
            var lmsCoeffs = Enumerable.Range(0, order).Select(currOrder => GetLmsCoefficient(order, stepIndex, currOrder)).ToArray();

            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors this.derivatives
            var revDerivatives = Enumerable.Reverse(this.Derivatives).ToList();

            // Create list of tuples from the lmsCoeffs and reversed derivatives
            var lmsCoeffsAndDerivatives = lmsCoeffs.Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative));

            // Create tensor for product of lmscoeffs and derivatives
            var lmsDerProduct = new Tensor<float>[this.Derivatives.Count()];

            for(int m = 0; m < lmsCoeffsAndDerivatives.Count(); m++)
            {
                var item = lmsCoeffsAndDerivatives.ElementAt(m);
                // Multiply to coeff by each derivatives to create the new tensors
                lmsDerProduct[m] = TensorHelper.MultipleTensorByFloat(item.derivative.ToArray(), (float)item.lmsCoeff, item.derivative.Dimensions.ToArray());
            }
            // Sum the tensors
            var sumTensor = TensorHelper.SumTensors(lmsDerProduct, new[] { 1, 4, 64, 64 });

            // Add the sumed tensor to the sample
            var prevSample = TensorHelper.AddTensors(latents.ToArray(), sumTensor.ToArray(), latents.Dimensions.ToArray());

            Console.WriteLine(prevSample[0]);
            return prevSample;

        }
    }
}
