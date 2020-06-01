
const svg = d3.select('#scatter-plot-svg');
const scatterPlotGroup = svg.append('g');
const scatterPlotTitle = scatterPlotGroup.append('text');
const xAxisGroup = scatterPlotGroup.append('g');
const xAxisLabel = xAxisGroup.append('text');
const yAxisGroup = scatterPlotGroup.append('g');
const yAxisLabel = yAxisGroup.append('text');

// @hack to work around “URL scheme must be ”http“ or ”https“ for CORS request.”
const data_location = "https://raw.githubusercontent.com/paul-tqh-nguyen/impact_of_attention/master/data/accuracy_vs_number_of_parameters.json"; // @todo remove this

const render = data => {

    const plotContainer = document.getElementById("scatter-plot");
    svg.style('background-color', 'white')
        .attr('width', plotContainer.clientWidth)
        .attr('height', plotContainer.clientHeight);

    const svg_width = parseFloat(svg.attr('width'));
    const svg_height = parseFloat(svg.attr('height'));
    
    // const getDatumAccuracy = datum => datum.test_accuracy;
    const getDatumLoss = datum => datum.test_loss;
    const getDatumParameterCount = datum => datum.number_of_parameters;

    const margin = {
        top: 80,
        bottom: 80,
        left: 120,
        right: 30,
    };
    
    const innerWidth = svg_width - margin.left - margin.right;
    const innerHeight = svg_height - margin.top - margin.bottom;
    const innerLineOpacity = 0.1;
    
    const xScale = d3.scaleLinear()
          .domain([0, d3.max(data, getDatumParameterCount)])
          .range([0, innerWidth]);
    
    const yScale = d3.scaleBand()
          .domain(data.map(getDatumLoss))
          .range([0, innerHeight])
          .padding(0.1);
    
    scatterPlotGroup.attr('transform', `translate(${margin.left}, ${margin.top})`);

    scatterPlotTitle
        .style('font-size', Math.min(20, innerWidth/40))
        .text("Test Accuracy vs Model Parameter Count")
        .attr('x', innerWidth * 0.225)
        .attr('y', -10);
    
    const yAxisTickFormat = number => d3.format('.3f')(number);
    yAxisGroup.call(d3.axisLeft(yScale).tickFormat(yAxisTickFormat).tickSize(-innerWidth));
    yAxisGroup.selectAll('.tick line').style('opacity', innerLineOpacity);
    yAxisLabel
        .style('font-size', 15)
        .attr('fill', 'black')
        .attr("transform", "rotate(-90)")
        .attr('y', -60)
        .attr('x', -innerHeight/3)
        .text('Mean Cross Entropy Loss');
    
    const xAxisTickFormat = number => d3.format('.3s')(number).replace(/G/,"B");
    xAxisGroup.call(d3.axisBottom(xScale).tickFormat(xAxisTickFormat).tickSize(-innerHeight))
          .attr('transform', `translate(0, ${innerHeight})`);
    xAxisGroup.selectAll('.tick line')
        .style('opacity', innerLineOpacity);
    xAxisLabel
        .style('font-size', 15)
        .attr('fill', 'black')
        .attr('y', margin.bottom * 0.75)
        .attr('x', innerWidth / 2)
        .text('Parameter Count');

    scatterPlotGroup.selectAll('circle').data(data)
        .enter()
        .append('circle') 
        .attr('y', datum => yScale(getDatumLoss(datum)))
        .attr('x', datum => xScale(getDatumParameterCount(datum)))
        .attr('r', 4);

};

const redraw = () => {
    d3.json(data_location)
        .then(data => {
            console.log(0);
            let extract_data = datum => {
                return {
                    test_loss: parseFloat(datum.test_loss),
                    test_accuracy: parseFloat(datum.test_accuracy),
                    number_of_parameters: parseInt(datum.number_of_parameters)
                };
            };
            console.log(1);
            let attention_data = data['attention'].map(extract_data);
            console.log(2);
            let plain_rnn_data = data['plain_rnn'].map(extract_data);
            console.log(3);
            render(attention_data, plain_rnn_data);
        }).catch(err => {
            console.error(err.message);
            return;
        });
};

redraw();
window.addEventListener("resize", redraw);

